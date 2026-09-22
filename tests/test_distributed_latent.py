import torch
import pytest

from medlatent.distributed.latent import (
    DecentralizedLatentNode, DistributedLatentProtocol, accumulation_steps_for, batched_pilot_loss, gossip_node_interfaces,
    interface_consensus_distance, load_decentralized_latent_checkpoint, load_distributed_latent_checkpoint,
    merge_kv_blocks, metropolis_mixing_weights, save_decentralized_latent_checkpoint,
    batched_relay_aggregate, batched_rollout, ring_two_hop_branches, save_distributed_latent_checkpoint,
    slice_kv_block, tiny_overfit,
)
from medlatent.modules import BoundaryEmbeddings, LatentDistiller
from medlatent.distributed.graph import path_graph, ring_graph


class _Output:
    def __init__(self, past_key_values, last_hidden_state, logits=None):
        self.past_key_values = past_key_values
        self.last_hidden_state = last_hidden_state
        self.logits = logits


class _Decoder(torch.nn.Module):
    def __init__(self, hidden=4, vocab=9):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.update = torch.nn.Linear(hidden, hidden, bias=False)
        self.output = torch.nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        hidden = self.update(values)
        if past_key_values is not None:
            hidden = hidden + past_key_values[0][0].mean(dim=(1, 2)).unsqueeze(1)
        key = hidden.unsqueeze(1)
        value = key + 0.1
        if past_key_values is not None:
            key = torch.cat([past_key_values[0][0], key], dim=2)
            value = torch.cat([past_key_values[0][1], value], dim=2)
        return _Output(((key, value),), hidden)


class _Model(_Decoder):
    base_model = property(lambda self: self)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return _Output(output.past_key_values, output.last_hidden_state, self.output(output.last_hidden_state))


def test_two_hop_rollout_reencodes_and_keeps_gradients():
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(ids)

    first = protocol.rollout(ids, mask)
    second = protocol.rollout(ids, mask, [first])
    assert first[0][0].shape[2] == 4  # BEGIN + m positions + END
    assert second[0][0].shape[2] == 4
    assert second[0][0].data_ptr() != first[0][0].data_ptr()
    assert merge_kv_blocks([first, first])[0][0].shape[2] == 8

    protocol.source_logits(ids, mask, [second]).sum().backward()
    assert protocol.distiller.projection[1].weight.grad.abs().sum() > 0
    assert protocol.distiller.latent_begin.grad.abs().sum() > 0
    assert protocol.boundary.begin.grad.abs().sum() > 0
    assert protocol.boundary.end.grad.abs().sum() > 0


def test_batched_greedy_generation_handles_variable_source_lengths():
    protocol = DistributedLatentProtocol(_Model(), LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    generated = protocol.generate(ids, mask, max_new_tokens=3, eos_token_id=None, pad_token_id=0)
    assert generated.shape == (2, 3)

    branch = protocol.rollout(torch.tensor([[1, 2], [3, 4]]), torch.ones((2, 2), dtype=torch.long))
    generated_with_branch = protocol.generate(ids, mask, [branch], max_new_tokens=2, eos_token_id=None, pad_token_id=0)
    assert generated_with_branch.shape == (2, 2)


# NOTE: Unused - Legacy
# def test_source_teacher_forcing_loss_uses_batched_branches():
#     model = _Model()
#     protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
#     ids = torch.tensor([[1, 2], [3, 4]])
#     mask = torch.ones_like(ids)
#     branches = two_hop_ring_blocks(protocol, leaf_ids=ids, leaf_mask=mask, relay_ids=ids, relay_mask=mask)
#     source_ids = torch.tensor([[1, 2]])
#     source_mask = torch.ones_like(source_ids)
#     target_ids = torch.tensor([[1, 2, 3]])
#     labels = target_ids.clone()
#     loss = protocol.source_loss(source_ids, source_mask, target_ids, labels, branches)
#     assert loss.ndim == 0 and torch.isfinite(loss)


def test_sliced_block_copies_storage_without_detaching():
    key = torch.randn(1, 1, 12, 4, requires_grad=True)
    value = torch.randn(1, 1, 12, 4, requires_grad=True)

    block = slice_kv_block(((key, value),), 4)

    assert block[0][0].untyped_storage().data_ptr() != key.untyped_storage().data_ptr()
    block[0][0].sum().backward()
    assert key.grad[:, :, -4:, :].abs().sum() > 0


def test_batched_local_and_relay_operator_detaches_children():
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(ids)
    local = batched_rollout(protocol, ids, mask, detach_children=False)
    relay = batched_relay_aggregate(protocol, ids, mask, [local])
    assert local[0][0].shape[:3] == relay[0][0].shape[:3] == (2, 1, 4)
    assert torch.autograd.grad(relay[0][0].sum(), local[0][0], allow_unused=True) == (None,)


def _pilot_tensors():
    source_ids = torch.tensor([[1, 2]])
    leaf_ids = torch.tensor([[1, 2], [3, 4]])
    return dict(
        source_ids=source_ids, source_mask=torch.ones_like(source_ids),
        target_ids=torch.tensor([[3, 4]]), target_labels=torch.tensor([[3, 4]]),
        leaf_ids=leaf_ids, leaf_mask=torch.ones_like(leaf_ids),
        relay_ids=leaf_ids, relay_mask=torch.ones_like(leaf_ids),
    )


def test_effective_batch_size_matches_medlatent_h_accumulation_semantics():
    assert accumulation_steps_for(batch_size=1, effective_batch_size=8) == 8
    assert accumulation_steps_for(batch_size=2, effective_batch_size=8) == 4
    assert accumulation_steps_for(batch_size=4, effective_batch_size=8) == 2
    with pytest.raises(ValueError, match="divisible"):
        accumulation_steps_for(batch_size=3, effective_batch_size=8)


def test_pilot_loss_only_updates_active_interface_parameters():
    protocol = DistributedLatentProtocol(_Model(), LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    first = batched_pilot_loss(protocol, use_relay=True, **_pilot_tensors())
    second = batched_pilot_loss(protocol, use_relay=True, **_pilot_tensors())
    assert torch.allclose(first, second)
    first.backward()

    assert protocol.distiller.projection[1].weight.grad.abs().sum() > 0
    assert protocol.boundary.begin.grad.abs().sum() > 0
    assert protocol.boundary.end.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in protocol.model.parameters())


def test_tiny_overfit_reduces_local_and_relay_losses_deterministically():
    torch.manual_seed(0)
    protocol = DistributedLatentProtocol(_Model(), LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    optimizer = torch.optim.AdamW([*protocol.distiller.parameters(), *protocol.boundary.parameters()], lr=0.05)
    result = tiny_overfit(protocol, optimizer, steps=20, **_pilot_tensors())

    assert result["local_final_loss"] < result["local_initial_loss"]
    assert result["relay_final_loss"] < result["relay_initial_loss"]


def test_fixed_route_and_checkpoint_metadata(tmp_path):
    assert ring_two_hop_branches(0, 10) == ((2, 1), (8, 9))
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    optimizer = torch.optim.AdamW(list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()))
    path = tmp_path / "distributed_latent.pt"
    save_distributed_latent_checkpoint(
        path, protocol=protocol, model_name="fake/model",
        route={"num_agents": 5, "R": 4, "k": 2, "m": 2, "pilot_hospital_ids": [0, 1, 2, 3, 4],
               "graph_edges": [[0, 1]], "split_paths": {"train": "train.json"}},
        optimizer=optimizer, training={"updates": 1, "tiny_overfit": {"local_final_loss": 1.0}},
    )
    payload = load_distributed_latent_checkpoint(path)
    assert payload["module_type"] == "DistributedLatentKV"
    assert payload["num_latents"] == 2
    assert payload["route"]["num_agents"] == 5
    assert payload["route"]["pilot_hospital_ids"] == [0, 1, 2, 3, 4]
    assert payload["training"]["tiny_overfit"]["local_final_loss"] == 1.0


def test_decentralized_nodes_have_isolated_interfaces_and_detached_relay():
    model = _Model()
    nodes = [DecentralizedLatentNode(index, model, 4, num_latents=2, device="cpu") for index in range(5)]
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(left is not right for left, right in zip(nodes[0].parameters, nodes[1].parameters))
    assert [tuple(parameter.shape for parameter in node.parameters) for node in nodes].count(
        tuple(parameter.shape for parameter in nodes[0].parameters)
    ) == len(nodes)

    ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(ids)
    source = torch.tensor([[3, 4]])
    target = torch.tensor([[3, 4]])
    leaf_block = nodes[0].protocol.rollout(ids, mask)
    assert nodes[0].train_step(
        mode="local", local_ids=ids, local_mask=mask, source_ids=source,
        source_mask=mask, target_ids=target, target_labels=target,
    ) >= 0
    assert nodes[1].train_step(
        mode="relay", local_ids=ids, local_mask=mask, source_ids=source,
        source_mask=mask, target_ids=target, target_labels=target,
        incoming=(leaf_block,),
    ) >= 0
    assert torch.autograd.grad(
        nodes[1].protocol.rollout(ids, mask, (leaf_block,), detach_incoming=True)[0][0].sum(),
        leaf_block[0][0], allow_unused=True,
    ) == (None,)


def test_metropolis_gossip_is_symmetric_and_reduces_consensus_distance():
    model = _Model()
    graph = ring_graph(3)
    nodes = {index: DecentralizedLatentNode(index, model, 4, num_latents=2) for index in range(3)}
    with torch.no_grad():
        for index, node in nodes.items():
            node.distiller.latent_begin.fill_(float(index))
            node.boundary.begin.fill_(float(index))
    weights = metropolis_mixing_weights(graph)
    assert all(abs(sum(row.values()) - 1.0) < 1e-6 for row in weights.values())
    assert all(weights[left][right] == weights[right][left] for left, row in weights.items() for right in row if left != right)
    before = interface_consensus_distance(nodes, graph)
    returned = gossip_node_interfaces(nodes, graph)
    after = interface_consensus_distance(nodes, graph)
    assert returned == weights
    assert after < before


def test_gossip_rejects_mismatched_interface_shapes():
    model = _Model()
    nodes = {
        0: DecentralizedLatentNode(0, model, 4, num_latents=2),
        1: DecentralizedLatentNode(1, model, 5, num_latents=2),
    }
    with pytest.raises(ValueError, match="identical shapes"):
        gossip_node_interfaces(nodes, path_graph(2))


def test_decentralized_checkpoint_keeps_node_and_gossip_metadata(tmp_path):
    model = _Model()
    graph = path_graph(2)
    nodes = {index: DecentralizedLatentNode(index, model, 4, num_latents=2) for index in range(2)}
    weights = gossip_node_interfaces(nodes, graph)
    path = tmp_path / "decentralized.pt"
    save_decentralized_latent_checkpoint(
        path, nodes=nodes, model_name="fake/model",
        route={"graph_edges": [list(edge) for edge in graph.edge_list()]},
        gossip={"rounds": 1, "mixing_weights": weights},
        training={"seed": 42, "local_steps": 1},
    )
    payload = load_decentralized_latent_checkpoint(path)
    assert payload["module_type"] == "DecentralizedLatentKV"
    assert sorted(payload["nodes"]) == [0, 1]
    assert payload["gossip"]["rounds"] == 1
    assert payload["training"]["seed"] == 42
