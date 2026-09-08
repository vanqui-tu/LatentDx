"""Lean distributed communication runtime for the M2 medical baseline."""

from .agent import AgentEpisodeState, AgentRuntime, PrivateKnowledgeStore
from .channels import StructuredChannel, TextChannel
from .episode import EpisodeEvent, EpisodeResult, SynchronousEpisodeEngine
from .graph import CommunicationGraph, complete_graph, erdos_renyi_graph, path_graph, ring_graph, stochastic_block_model_graph, watts_strogatz_graph
from .medical import (
    HospitalPrivateStore, MedicalDatasetSplits, MedicalEpisode,
    MedicalQuery, MedicalQueryRecord, MedicalRetrievedRecord, RetrievalAuditEvent,
    build_medical_agents, load_hospital_private_stores, load_medical_dataset_splits,
    load_medical_split, prediction_matches_target, sample_balanced_sources,
)
from .medical_baselines import MedicalBaselineKind, MedicalBaselineResult, run_medical_baseline
from .messages import MessageEnvelope, MessageKind, ProposalPayload, RequestPayload, TextProposalPayload, TextRequestPayload
from .textmas import TextGeneration, TextGenerator, TransformersTextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer

__all__ = [
    "AgentEpisodeState", "AgentRuntime", "PrivateKnowledgeStore",
    "CommunicationGraph", "complete_graph", "path_graph", "ring_graph", "erdos_renyi_graph", "watts_strogatz_graph", "stochastic_block_model_graph",
    "MessageEnvelope", "MessageKind", "ProposalPayload", "RequestPayload", "TextProposalPayload", "TextRequestPayload",
    "EpisodeEvent", "EpisodeResult", "SynchronousEpisodeEngine", "StructuredChannel", "TextChannel",
    "HospitalPrivateStore", "MedicalDatasetSplits", "MedicalEpisode", "MedicalQuery", "MedicalQueryRecord",
    "MedicalRetrievedRecord", "RetrievalAuditEvent", "build_medical_agents", "load_hospital_private_stores",
    "load_medical_dataset_splits", "load_medical_split", "prediction_matches_target", "sample_balanced_sources",
    "MedicalBaselineKind", "MedicalBaselineResult", "run_medical_baseline",
    "TextGeneration", "TextGenerator", "TransformersTextGenerator", "build_agent_prompt", "build_host_prompt", "parse_textmas_answer",
]
