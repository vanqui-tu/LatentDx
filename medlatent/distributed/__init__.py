"""Distributed communication primitives for private-knowledge diagnosis."""

from .agent import AgentEpisodeState, AgentRuntime
from .aggregation import AggregatedProposal, majority_vote, max_confidence, mean_score, require_all_evidence
from .baselines import BaselineKind, BaselineResult, run_baseline
from .channels import StructuredChannel
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
from .routing import DirectNeighborRouter, FloodUnvisitedRouter, RandomKRouter, Router
from .results import EpisodeLogRecord

__all__ = [
    "AckPayload",
    "AggregatedProposal",
    "AgentEpisodeState",
    "AgentRuntime",
    "BaselineKind",
    "BaselineResult",
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
    "StructuredChannel",
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
    "sample_balanced_sources",
]
