"""Lean distributed communication runtime for the M2 medical baseline."""

from .agent import AgentEpisodeState, AgentRuntime, PrivateKnowledgeStore
from .channels import StructuredChannel, TextChannel
from .episode import EpisodeEvent, EpisodeResult, SynchronousEpisodeEngine
from .graph import (CommunicationGraph, complete_graph, erdos_renyi_graph, extend_graph, path_graph, ring_graph,
                    seeded_shortcut_ring_graph, shortcut_ring_graph, stochastic_block_model_graph,
                    validate_graph_constraints, watts_strogatz_graph)
from .medical import (
    HospitalPrivateStore, MedicalAgentBatch, MedicalDatasetSplits, MedicalEpisode, Q4R6Paths,
    MedicalQuery, MedicalQueryRecord, MedicalRetrievedRecord, RetrievalAuditEvent,
    build_medical_agents, load_hospital_private_stores, load_medical_dataset_splits,
    load_medical_split, load_q4r6_dataset, load_q4r6_episodes, prediction_matches_target, q4r6_paths,
    sample_balanced_sources, select_pilot_hospital_ids, build_agent_query_batch, build_agent_prompt_batch, build_agent_retrieval_batch,
)
from .medical_baselines import MedicalBaselineKind, MedicalBaselineResult, run_medical_baseline
from .messages import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload, TextProposalPayload, TextRequestPayload
from .textmas import TextGeneration, TextGenerator, TransformersTextGenerator, VllmTextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer
from .latent import (
    DecentralizedLatentNode, DistributedLatentProtocol, KVBlock, LatentKVProtocol, NodeLocalLatentTrainer, accumulation_steps_for, batched_pilot_loss, batched_relay_aggregate, batched_rollout,
    detach_kv_block, gossip_node_interfaces, interface_consensus_distance, kv_wire_bytes,
    load_decentralized_latent_checkpoint, load_distributed_latent_checkpoint, metropolis_mixing_weights,
    build_two_hop_route, canonical_two_hop_losses, graph_broadcast_tree, graph_two_hop_branches, merge_kv_blocks, ring_two_hop_branches, save_distributed_latent_checkpoint,
    save_decentralized_latent_checkpoint, select_kv_rows, slice_kv_block, tiny_overfit, train_distributed_latent_real
)

__all__ = [
    "AgentEpisodeState", "AgentRuntime", "PrivateKnowledgeStore",
    "CommunicationGraph", "complete_graph", "extend_graph", "path_graph", "ring_graph", "shortcut_ring_graph", "seeded_shortcut_ring_graph",
    "validate_graph_constraints", "erdos_renyi_graph", "watts_strogatz_graph", "stochastic_block_model_graph",
    "MessageEnvelope", "MessageKind", "ProposalPayload", "RequestPayload", "TextProposalPayload", "TextRequestPayload",
    "EpisodeEvent", "EpisodeResult", "SynchronousEpisodeEngine", "StructuredChannel", "TextChannel",
    "HospitalPrivateStore", "MedicalAgentBatch", "MedicalDatasetSplits", "MedicalEpisode", "Q4R6Paths", "MedicalQuery", "MedicalQueryRecord",
    "MedicalRetrievedRecord", "RetrievalAuditEvent", "build_medical_agents", "load_hospital_private_stores",
    "load_medical_dataset_splits", "load_medical_split", "load_q4r6_dataset", "load_q4r6_episodes", "q4r6_paths",
    "prediction_matches_target", "sample_balanced_sources", "select_pilot_hospital_ids", "build_agent_query_batch", "build_agent_prompt_batch", "build_agent_retrieval_batch",
    "MedicalBaselineKind", "MedicalBaselineResult", "run_medical_baseline",
    "TextGeneration", "TextGenerator", "TransformersTextGenerator", "VllmTextGenerator", "build_agent_prompt", "build_host_prompt", "parse_textmas_answer",
    "DistributedLatentProtocol", "DecentralizedLatentNode", "NodeLocalLatentTrainer", "LatentKVProtocol", "KVBlock", "merge_kv_blocks", "slice_kv_block", "select_kv_rows", "detach_kv_block",
    "batched_rollout", "batched_relay_aggregate", "batched_pilot_loss", "tiny_overfit", "accumulation_steps_for", "kv_wire_bytes",
    "ring_two_hop_branches", "graph_two_hop_branches", "build_two_hop_route", "canonical_two_hop_losses", "graph_broadcast_tree",
    "metropolis_mixing_weights", "gossip_node_interfaces", "interface_consensus_distance",
    "save_distributed_latent_checkpoint", "load_distributed_latent_checkpoint",
    "save_decentralized_latent_checkpoint", "load_decentralized_latent_checkpoint",
    "train_distributed_latent_real",
]
