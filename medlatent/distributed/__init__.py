"""Distributed communication primitives for private-knowledge diagnosis."""

from .agent import AgentEpisodeState, AgentRuntime
from .aggregation import AggregatedProposal, majority_vote, max_confidence, mean_score, require_all_evidence
from .baselines import BaselineKind, BaselineResult, run_baseline
from .medical_baselines import MedicalBaselineKind, MedicalBaselineResult, MedicalTraceEvent, run_medical_baseline
from .textmas import TextGeneration, TextGenerator, TransformersTextGenerator, build_agent_prompt, build_host_prompt, parse_textmas_answer
from .channels import StructuredChannel, TextChannel, find_raw_substring_leaks
from .config import DistributedBaselineConfig, TopologyConfig
from .episode import EpisodeEvent, EpisodeResult, SynchronousEpisodeEngine
from .fixtures import ComplementaryEvidenceFixture, complementary_evidence_fixture
from .graph import (
    CommunicationGraph,
    DegreeStatistics,
    complete_graph,
    path_graph,
    random_regular_graph,
    ring_graph,
    star_graph,
)
from .messages import (
    AckPayload,
    CommunicationCost,
    EvidencePayload,
    MessageEnvelope,
    MessageKind,
    ProposalPayload,
    RequestPayload,
    TextProposalPayload,
    TextRequestPayload,
)
from .medical import (
    HospitalPrivateStore,
    MedicalDatasetSplits,
    MedicalEpisode,
    MedicalQuery,
    MedicalQueryRecord,
    MedicalRetrievedRecord,
    RetrievalAuditEvent,
    build_medical_agents,
    load_hospital_private_stores,
    load_medical_dataset_splits,
    load_medical_split,
    sample_balanced_sources,
)
from .store import PrivateKnowledgeStore, SyntheticKnowledgeRecord, SyntheticKnowledgeStore
from .routing import DirectNeighborRouter, FloodUnvisitedRouter, PublicExpertiseRouter, RandomKRouter, Router
from .results import EpisodeLogRecord

__all__ = [
    "AckPayload",
    "AggregatedProposal",
    "AgentEpisodeState",
    "AgentRuntime",
    "BaselineKind",
    "BaselineResult",
    "MedicalBaselineKind",
    "MedicalBaselineResult",
    "MedicalTraceEvent",
    "TextGeneration",
    "TextGenerator",
    "TransformersTextGenerator",
    "CommunicationCost",
    "CommunicationGraph",
    "ComplementaryEvidenceFixture",
    "DegreeStatistics",
    "DistributedBaselineConfig",
    "EpisodeEvent",
    "EpisodeResult",
    "EpisodeLogRecord",
    "EvidencePayload",
    "MessageEnvelope",
    "MessageKind",
    "HospitalPrivateStore",
    "MedicalDatasetSplits",
    "MedicalEpisode",
    "MedicalQuery",
    "MedicalQueryRecord",
    "MedicalRetrievedRecord",
    "PrivateKnowledgeStore",
    "ProposalPayload",
    "RequestPayload",
    "Router",
    "RetrievalAuditEvent",
    "RandomKRouter",
    "DirectNeighborRouter",
    "FloodUnvisitedRouter",
    "PublicExpertiseRouter",
    "StructuredChannel",
    "TextChannel",
    "TextProposalPayload",
    "TextRequestPayload",
    "find_raw_substring_leaks",
    "SynchronousEpisodeEngine",
    "SyntheticKnowledgeRecord",
    "SyntheticKnowledgeStore",
    "TopologyConfig",
    "complementary_evidence_fixture",
    "build_medical_agents",
    "complete_graph",
    "path_graph",
    "random_regular_graph",
    "ring_graph",
    "star_graph",
    "majority_vote",
    "max_confidence",
    "mean_score",
    "load_hospital_private_stores",
    "load_medical_dataset_splits",
    "load_medical_split",
    "require_all_evidence",
    "run_baseline",
    "run_medical_baseline",
    "build_agent_prompt",
    "build_host_prompt",
    "parse_textmas_answer",
    "sample_balanced_sources",
]
