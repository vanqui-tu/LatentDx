import torch

from medlatent.distributed.latent import DistributedLatentProtocol, merge_kv_blocks, slice_kv_block
from medlatent.modules import BoundaryEmbeddings, LatentDistiller


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


def test_sliced_block_copies_storage_without_detaching():
    key = torch.randn(1, 1, 12, 4, requires_grad=True)
    value = torch.randn(1, 1, 12, 4, requires_grad=True)

    block = slice_kv_block(((key, value),), 4)

    assert block[0][0].untyped_storage().data_ptr() != key.untyped_storage().data_ptr()
    block[0][0].sum().backward()
    assert key.grad[:, :, -4:, :].abs().sum() > 0
