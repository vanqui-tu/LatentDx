"""Synthetic tasks that require complementary private evidence."""

from __future__ import annotations

from dataclasses import dataclass

from .agent import AgentRuntime
from .graph import CommunicationGraph, path_graph
from .store import SyntheticKnowledgeRecord, SyntheticKnowledgeStore


@dataclass(frozen=True, slots=True)
class ComplementaryEvidenceFixture:
    graph: CommunicationGraph
    query: tuple[str, ...]
    source_id: int
    target_label: str
    required_evidence_labels: tuple[str, ...]
    necessary_agent_ids: tuple[int, ...]

    def build_agents(self) -> dict[int, AgentRuntime]:
        stores = {
            0: SyntheticKnowledgeStore(()),
            1: SyntheticKnowledgeStore((SyntheticKnowledgeRecord("private-a", self.query, "EVIDENCE:A"),)),
            2: SyntheticKnowledgeStore((SyntheticKnowledgeRecord("private-b", self.query, "EVIDENCE:B"),)),
        }
        return {agent_id: AgentRuntime(agent_id, stores[agent_id]) for agent_id in self.graph.agent_ids}


def complementary_evidence_fixture(graph: CommunicationGraph | None = None) -> ComplementaryEvidenceFixture:
    topology = graph or path_graph(3)
    if topology.num_agents != 3:
        raise ValueError("complementary evidence fixture requires exactly three agents")
    return ComplementaryEvidenceFixture(
        graph=topology,
        query=("signal-a", "signal-b"),
        source_id=0,
        target_label="DX:combined",
        required_evidence_labels=("EVIDENCE:A", "EVIDENCE:B"),
        necessary_agent_ids=(1, 2),
    )
