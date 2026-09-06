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
    "PrivateKnowledgeStore",
    "ProposalPayload",
    "RequestPayload",
    "Router",
    "RandomKRouter",
    "DirectNeighborRouter",
    "FloodUnvisitedRouter",
    "StructuredChannel",
    "SynchronousEpisodeEngine",
    "SyntheticKnowledgeRecord",
    "SyntheticKnowledgeStore",
    "TopologyConfig",
    "complementary_evidence_fixture",
    "complete_graph",
    "path_graph",
    "random_regular_graph",
    "ring_graph",
    "star_graph",
    "majority_vote",
    "max_confidence",
    "mean_score",
    "require_all_evidence",
    "run_baseline",
]
