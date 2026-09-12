# Distributed Communication for Private-Knowledge Diagnosis

> Working research and implementation plan. Last updated: 2026-09-12.
>
> Current scope: Phase 1 supports fixed, simple, undirected graphs only.
> Directed graphs, dynamic topology, node churn, and network failures are
> deferred. The task construction, threat model, and learned latent method
> remain research decisions.

## 1. Goal

Evolve LatentDx from fixed centralized fan-in into query-initiated inference
over a fixed communication graph. A query can arrive at any source agent. Each
agent owns a private store, may communicate only with graph neighbors, and may
contact at most `k` neighbors per round. The source should produce the final
diagnosis after at most `R` rounds.

The first objective is not to learn the protocol. It is to build a small,
reproducible simulator in which topology, routing, message representation, and
aggregation can be varied independently. Simple non-learned baselines should
run on CPU before multi-round latent communication is introduced.

The intended research progression is:

1. fixed networks and deterministic episode semantics;
2. local, structured, and text communication baselines;
3. multi-hop routing and GNN-like aggregation baselines;
4. fixed-routing latent communication;
5. learned latent messages, routing, stopping, and aggregation;
6. privacy, scalability, and failure/adversary evaluation.

## 2. Working Formulation

Let `G = (V, E)` be a fixed undirected graph for one inference episode, with
`V = [N]`. Phase 1 requires

```text
A[i, j] = A[j, i]
A[i, i] = 0
A[i, j] = 1  <=>  i and j may directly exchange messages.
```

Agent `i` owns private knowledge `K_i`. A query `q` and its source `s` define
an episode `e = (q, s, G, {K_i}, R, k)`. Initially only `s` is active and
computes

```text
(h_s^0, y_hat_s^0) = F_s(q, K_s).
```

At round `r`, active agent `i`:

1. reads messages delivered at the end of round `r - 1`;
2. updates its local episode state from its private store and inbox;
3. selects at most `k` outgoing neighbors allowed by `A`;
4. emits messages through the configured channel; and
5. optionally emits a proposal or reply intended to return to the source.

For the first implementation, delivery is reliable and synchronous: messages
sent during round `r` are visible in round `r + 1`. Each `(agent, query)` has
isolated state. Duplicate message IDs are ignored. An episode terminates after
`R` rounds or when the source's stopping rule fires. The source, rather than a
new privileged host, owns the final prediction.

Self-loops are disabled. One undirected edge permits sends in both directions,
but each send is separately logged and charged. `k` is a per-sender,
per-round fan-out bound, not a global message bound. With next-round delivery,
a request to a one-hop neighbor followed by its reply requires at least two
rounds. Under request-return routing, evidence at distance `d` cannot influence
the source in fewer than `2d` rounds.

### 2.1 Privacy boundary for the MVP

Raw store items never leave their owner. Baseline messages may contain derived
predictions, scores, requests, or generated summaries, but never a serialized
record or a direct store reference. This is a data-flow constraint, not a
formal privacy guarantee.

Text summaries may reveal private attributes even when they are not verbatim
records. They should therefore be treated as a utility baseline with measured
leakage, not as a privacy-preserving method. A raw-record broadcast can be used
only as a clearly marked, non-compliant oracle upper bound and must not be
reported as a valid decentralized-private baseline.

## 3. Separate the Experimental Axes

Keep four axes conceptually distinct so comparisons remain interpretable; they
do not each require an interface, registry, or class hierarchy:

| Axis | Responsibility | Examples |
| --- | --- | --- |
| Topology | Who can directly send to whom | ring, random regular, small-world |
| Router | Which allowed edges are selected now | broadcast, random-`k`, score-top-`k` |
| Channel | How a semantic payload is represented and charged | structured, text, latent |
| Aggregator | How received messages update local state | vote, mean, max, attention |

The episode engine owns round scheduling, activation, inboxes, and budgets. An
agent owns local retrieval/inference and must never receive another agent's
store object. For M2 the topology, router, and aggregator are fixed, so simple
functions in the baseline module are sufficient; only the channel changes.

## 4. Network Initialization

### 4.1 Canonical graph object

Use one small `CommunicationGraph` value with:

```text
num_agents: int
adjacency: bool[N, N]
```

It exposes only `neighbors(i)`, `has_edge(i, j)`, `shortest_path_length(i, j)`,
and `edge_list()`. A dense Boolean matrix is adequate for the current five-node
experiment. The canonical runner records the realized edge list; a graph
metadata schema, hash, and load/save API are unnecessary at this stage.

### 4.2 Active topologies

Keep `ring_graph` for the primary experiment, `complete_graph` for the
all-hospital reference, and `path_graph` for the two-hop test. Star,
random-regular, Erdos-Renyi, Watts-Strogatz, stochastic-block, scale-free, and
external graph support are deferred. They are easy to add after topology itself
becomes an experimental variable.

### 4.3 Validation and episode feasibility

Initialization should reject:

- non-square or non-binary adjacency matrices;
- illegal node IDs or self-loops when disabled;
- asymmetric adjacency matrices; and
- sends to a non-neighbor or fan-out above `k`.

The fundamental feasibility condition is more specific than connectivity. If
`I*(q)` is the set of evidence-bearing agents, the graph and round budget must
permit evidence to influence the source. In the initial request-return
protocol, an evidence-bearing agent at shortest-path distance `d` requires at
least `2d` rounds, before accounting for fan-out and routing failures.
The `R=2` versus `R=4` ring comparison is sufficient to expose out-of-budget
two-hop evidence in M2; no general connectivity policy is needed.

## 5. Message and Channel Contract

### 5.1 Transport envelope

All representations share a small in-memory envelope:

```text
MessageEnvelope
  message_id
  episode_id / query_id
  round_sent
  sender_id / receiver_id
  kind: REQUEST | PROPOSAL
  payload
  wire_bytes
```

The envelope enables deduplication, causal tracing, reply routing, and precise
communication accounting. It does not need a general JSON deserializer because
the current runtime is in-process. Event logs should not contain private
payload contents.

### 5.2 Channel interface

A channel builds one bounded payload from local evidence and reports its size:

```text
build(local_state) -> payload
wire_bytes(payload) -> int
```

The channel does not pick recipients, mutate the graph, or schedule delivery.

Initial channels:

1. `StructuredChannel`: candidate label and retrieval score.
2. `TextChannel`: bounded generated diagnostic summary/request. It must apply a
   token limit and record prompt/completion tokens and UTF-8 bytes.
3. `LatentChannel`: fixed `m x d` tensor or model KV block plus dtype/shape/model
   metadata. Add only after the structured/text engine is correct.

Report sends and transmitted bytes. Text tokens and latent positions can be
included later as representation-specific diagnostics, but they do not require
a shared accounting framework.

## 6. Baseline Protocol Ladder

This section is a method catalog, not the active implementation checklist.
Section 8 currently activates only B0, B1, B2-structured, and B2-text. B3-B6
remain available ideas and must not be expanded until the fixed M2 result is
known. Every active baseline uses the same episode engine and obeys the same
`A`, `k`, and `R`, unless explicitly labeled an oracle.

### B0. Local only

The source diagnoses using `K_s` only. No messages. This establishes how much
the task actually requires collaboration and should be stratified by whether
the source owns relevant evidence.

### B1. One-hop structured consultation

The source requests up to `k` neighbors. Each consulted agent retrieves
locally and returns a bounded structured proposal. The source aggregates by
majority vote or mean normalized score. Use uniform/random neighbor selection
with a fixed seed first.

Because the graph is undirected, a consulted neighbor may reply over the same
edge in the following round. Request and reply remain two distinct sends and
both count toward their respective sender's per-round budget.

### B2. Multi-hop flooding with deduplication

Forward a request to up to `k` unvisited neighbors until TTL/round budget
expires. The first-arrival parent tree supplies the return path for
evidence/proposals. This baseline tests reachability and aggregation without
learned routing.

### B3. Random-`k` / random-walk routing

Select one or `k` neighbors uniformly at each active node. Compare performance
and relevant-agent recall against flooding at a lower communication cost.

### B4. Heuristic relevance routing

Each agent advertises only a public, coarse expertise descriptor. Route to the
top-`k` neighbors by query-to-expertise similarity, without inspecting their
stores. This is the first meaningful agent-discovery baseline. An oracle
router that knows `I*(q)` is useful only as an upper bound.

### B5. Text collaboration

Repeat B1-B4 with bounded text requests/evidence summaries and source-side LLM
aggregation. Keep decoding deterministic (`temperature = 0`) initially. This
measures the benefit and leakage of an expressive channel before latent
communication.

### B6. GNN-like fixed message passing

Represent each agent with a fixed-size state and apply shared updates for `R`
rounds:

```text
m_ij^r = M(h_i^r, q, edge_ij)
bar_m_j^r = AGG({m_ij^r : i -> j is selected})
h_j^(r+1) = U(h_j^r, bar_m_j^r, local_j)
```

Start with non-learned mean/max/vote aggregation, then a small trainable MLP or
attention aggregator on frozen local features. This provides a clean bridge
to learned latent communication without involving full LLM fine-tuning.

### Diagnostic oracles

Keep the following out of the main compliant baseline table:

- all agents consulted regardless of `G`, `k`, or `R`;
- oracle relevant-agent routing;
- centralized host aggregation; and
- raw-record sharing.

They diagnose upper bounds and failure sources, but violate one or more target
constraints.

## 7. Adapting the Current LatentDx Codebase

The current `MedLatentDiagnosisDataset` retrieves one case from every hospital
and constructs all hospital prompts before the model forward pass. The current
MedLatent-H/X path then concatenates all hospital blocks into one host prefix.
That behavior is centralized and should remain available as a legacy/oracle
baseline, not become the distributed runtime.

The distributed path needs three data boundaries:

1. `QueryRecord`: public episode query and target used for evaluation;
2. `PrivateKnowledgeStore`: one store per agent, exposing only local
   `retrieve(query)` to its owner; and
3. `AgentRuntime`: local model/retriever/state plus public neighbors, with no
   container holding all stores passed into its inference method.

For the first medical smoke experiment, map each existing hospital shard to an
agent and sample source agents deterministically. The source sees the query and
its own shard. Other agents see the query only after receiving a request. The
final diagnosis is generated at the source. This reuses the reproduced data
while testing the new system contract.

This dataset is not sufficient evidence for the final research claim. The
existing task may often reduce to matching a single retrieved case. We should
add episode annotations for an approximate `I*(q)` and measure whether answers
change when necessary agents are removed. A later benchmark should make
complementary multi-agent evidence and multi-hop composition unavoidable by
construction.

### 7.1 Lean code rule

The existing `medlatent/distributed/` substrate is sufficient. From M2 onward,
do not add a module merely to represent an experiment schema, a seed bundle, a
summary table, or one alternative routing policy. Prefer a plain dictionary in
the experiment script and a small helper in an existing module.

Add a new file only when it contains an independently testable research
algorithm that cannot fit cleanly in an existing file. One canonical runner is
preferred over a generic sweep API plus several wrappers. Configuration should
contain only values changed in the current experiment; constants that define
the experiment belong together in the runner.

For every proposed abstraction, ask: does it enable a comparison in the next
result table, or is it needed by both text and latent implementations? If the
answer is no, leave it out. Research reproducibility requires saving the exact
command, graph, seed, model, and per-query output; it does not require turning
each of those fields into a framework type.

### 7.2 Lean target after pruning

`bd5d825` is a behavioral checkpoint, not a structure that must be preserved.
The target is the smallest code path that runs B0/B1/B2 with structured, text,
and later latent payloads:

```text
medlatent/distributed/
  graph.py               graph + ring/complete/path helpers
  messages.py            request/proposal envelopes and byte cost
  channels.py            structured/text; small latent KV payload helper
  agent.py               query-scoped local state
  episode.py             synchronous scheduler and event trace
  medical.py             hospital stores, query loading, source assignment
  medical_baselines.py   B0/B1/B2 and mean-score aggregation
  textmas.py             Transformers adapter and prompts
  latent.py              fixed-route latent training/inference path

scripts/
  run_distributed_medical.py
```

Keep these files separate because they are also the natural boundary for the
latent path. Do not merge them merely to reduce file count. Conversely, remove
entire experiment-only layers rather than keeping generalized APIs with no
active consumer.

Do not import the neighboring `DecentralizedMAS` repository as a runtime
dependency. Its fixed-topology strategy and queue-based transport are useful
references, but its message schema is text-only and its autonomous loop does
not yet encode query-scoped synchronous inference. Reusing its high-level
ideas is safer than coupling two evolving repositories.

## 8. Lean M2 Experiment

M2 has one purpose: determine whether remote evidence over a sparse multi-hop
network improves diagnosis enough to justify building latent communication.
It is not a topology, routing, or statistics framework phase.

### 8.1 Canonical setting

Use one fixed setting first:

```text
data: data_original_skewed/test.json (all 401 cases)
private stores: the five data_original_skewed hospital shards
N: 5
hospital-to-node mapping: identity
source assignment: existing balanced assignment, seed 42
graph: ring
R: 4
k: 2
router: flood-unvisited
retrieval: current HPO top-1
aggregation: current mean-score rule
```

On a five-node ring, every node is at most two hops away, and `R=4` permits a
request-return path from the farthest node. This is the smallest setting that
actually exercises sparse multi-hop communication without adding arbitrary
graph choices.

Run only these comparisons:

1. `B0 local`: source retrieval only;
2. `B1 one-hop structured`: checks whether immediate neighbors suffice;
3. `B2 multi-hop structured`: fixed ring/flooding reference;
4. `B2 multi-hop text`: exactly the same episodes and route as structured.

For TextMAS, start with the first fixed 50 cases from the same ordered test
split and one frozen model. Expand to 401 only if the pilot changes conclusions
and runtime is acceptable. Hugging Face `transformers` remains the reference
backend because the latent path will need its hidden states/KV cache. For quick
remote inference, the text baseline may call an external vLLM OpenAI-compatible
server (`/v1/chat/completions`); vLLM is intentionally not a repository
dependency. The adapter sends standard `temperature=0`, `top_p=1`, `seed=42`,
and Qwen `chat_template_kwargs.enable_thinking=false`. Load the
model once, use greedy decoding and record its exact path/name. Generation
caching is an optimization, not an M2 prerequisite.

### 8.2 Minimal outputs

Write one plain JSONL row per `(case, method)` with only:

```text
case_id, source_id, method, prediction, target,
contacted_agent_ids, messages, wire_bytes
```

Write one summary JSON containing per-method accuracy, mean messages, and mean
bytes. A direct retrieval scan in the runner should also report three dataset
facts once: fraction with a source-local gold hit, fraction with a remote gold
hit, and fraction with no gold hit anywhere. These are diagnostics, not another
audit abstraction.

No alias registry, bootstrap utility, config hash, comparison ID, graph hash,
hospital permutation, per-agent cost table, or five-way seed object is required
for M2. The fixed command, Git commit, data directory, seed 42, and saved graph
are enough to reproduce this stage.

### 8.3 Two small ablations

Only after the canonical run succeeds, run:

- `R=2` on the same ring to isolate one-hop versus two-hop availability;
- complete graph with `R=2`, `k=4` as an all-hospital upper reference.

Do not sweep star/path/random-regular, multiple node permutations, multiple
seed families, or public-expertise/oracle routing in M2. Routing research can be
added later if fixed multi-hop communication first proves useful.

### 8.4 Go/no-go criterion

Proceed to latent communication when B2 improves over B0 on the full skewed
test set and the gain is concentrated in cases with useful remote evidence.
B1 versus B2 should also reveal whether the second hop matters. Exact effect
size is empirical; save paired per-case predictions so it can be inspected
without implementing a statistics subsystem.

If B2 does not improve, or almost every correct B2 case is already correct at
B0, stop and redesign the data/task partition. More routers, graph types, and
result schemas will not repair a task that does not require collaboration.

### 8.5 One 10-hospital robustness check

Structured results on all 401 cases:

| Setting | B0 | B1 | B2 |
| --- | ---: | ---: | ---: |
| Ring `R=4,k=2` | 38.65% | 65.59% | 72.32% |
| Ring `R=10,k=2` | 38.65% | 65.59% | 73.57% |
| Complete `R=2,k=9` | 38.65% | 73.57% | 73.57% |

```text
outputs_distributed/structured_10_ring_r4_k2
outputs_distributed/structured_10_ring_r10_k2
outputs_distributed/structured_10_complete_r2_k9
```

The structured check is complete. Run only the matched 50-case text pilot
before M3; defer additional topology experiments.

## 9. Deferred Evaluations

Macro-F1, confidence intervals, expertise routing, oracle routing, topology
sweeps, load balance, privacy attacks, and scaling beyond ten hospitals are
reasonable later evaluations. They are explicitly outside the M2 critical
path. Add one only when the canonical result exposes a concrete question that
the additional evaluation answers.

## 10. GNN Connection and Limits

The setting naturally matches message-passing neural networks: agents are
nodes, allowed communication links are edges, messages are edge values,
aggregation is permutation-invariant neighborhood reduction, and `R` bounds
the receptive field to at most `R` hops. This perspective gives useful tools:

- shared local update/message functions across agents;
- mean/sum/max/attention aggregators;
- degree normalization and residual state updates;
- graph minibatching and masks for variable topologies;
- over-squashing diagnostics when distant evidence must cross narrow cuts;
- over-smoothing diagnostics as rounds increase; and
- graph rewiring/routing as learned edge selection.

The analogy is not exact. Agents perform retrieval and language reasoning,
messages may be discrete or model-specific KV tensors, routing is conditional
and budgeted, and privacy leakage matters. We should use GNN machinery for the
communication computation graph without reducing the whole problem to a
standard node-prediction benchmark.

Useful early hypotheses to test are:

- small-world topology improves accuracy per transmitted byte on queries whose
  relevant knowledge is distributed across communities;
- adaptive top-`k` routing alleviates over-squashing better than increasing
  message width alone;
- residual/gated state updates retain local evidence better over many rounds;
- latent messages offer a better utility-bandwidth-leakage frontier than text,
  but are not private by construction.

These are hypotheses, not assumed conclusions.

## 11. Learned Latent Communication (M3)

M3 is a fresh trainable experiment, not a patch to the deleted prototype. Use
the fixed M2 setting: `N=10`, ring, `R=4`, `k=2`, the same private stores,
retrieval, query split, and B2 parent routes. Routing and aggregation are fixed;
only the latent interface is learned.

### 11.1 Transport contract

Use a contextualized **KV block** in the frozen backbone's native cache format,
as in MedLatent-H. Do not send raw hidden states or text. One block has exactly
`m` latent positions and contains the key/value tensors for those positions at
every decoder layer. All agents use one backbone/model family, so no projector
or cross-family adapter is needed.

During training, the block is an in-memory differentiable object (a tuple of
per-layer key/value tensors). Do not serialize it or detach it between hops.
During evaluation, it may be detached for logging and its wire cost is the
serialized tensor byte count plus a small fixed metadata header.

### 11.2 Per-agent computation

Each active agent has one private local prompt `x_i` built from the query and
its own top-1 retrieved case. Let `B_in` be the KV cache returned by the parent,
or empty at the source. The agent runs the frozen decoder with `B_in` as past
context and `x_i` as new input, then appends learned latent markers and rolls
out `m` latent positions using the new shared `LatentDistiller`:

```text
past = B_in
past = Decoder(past, x_i)
past = Decoder(past, BEGIN)
repeat m times:
    z = Distiller(previous_hidden)
    past = Decoder(past, z)
past = Decoder(past, END)
B_out = last (m + BEGIN + END) KV positions from past
```

`B_out` is newly computed at every hop and includes the agent's local evidence
plus the incoming context. The intermediate agent never forwards `B_in`
unchanged. A relay performs in-network aggregation: it consumes child block(s)
and its own local evidence, emits one new `B_out`, and returns only that summary
along the parent path. The source therefore receives one final summary per
branch, not every node-local block. In the N=10 ring each one-hop branch relay
has at most one outward child, so this is a sequential re-encoding. If a future
topology gives a relay multiple children, concatenate them in stable child order
before re-encoding; no learned set aggregator is part of M3. The source
concatenates the final summaries from the two ring branches in deterministic
branch order, similar to the original MedLatent host concatenation.

### 11.3 Train from new parameters

Load only a frozen base model and tokenizer. Initialize a new shared
`LatentDistiller` and new learned `BEGIN`/`END` embeddings; old MedLatent-H/X
checkpoints are optional references only and must not be required inputs. The
only trainable parameters are this distiller and the two boundary embeddings.

For each training episode, execute the complete fixed B2 route, construct the
source context from the returned blocks, and teacher-force the gold diagnosis.
Use ordinary diagnosis cross-entropy:

```text
L = CE(gold_answer | source_private_query, returned_KV_blocks)
```

Keep the backbone in `eval()` and freeze all backbone parameters, but preserve
autograd through every decoder call and every in-memory KV block. No `.detach()`
or `torch.no_grad()` is allowed around the training forward pass. The training
checkpoint contains the new distiller, boundaries, model identifier, `m`, route
settings, and optimizer/training metadata. It is a new checkpoint even when the
base model is the same one used by MedLatent.

Run one episode at a time and process the two ring branches sequentially; do
not batch all agents or retain each relay's full prompt cache. The final KV
slice must be cloned, not detached, so its small storage survives while its
gradient path remains intact. Checkpoint each decoder call while gradients are
enabled to recompute frozen-backbone activations during backward. This trades
compute for bounded VRAM and still retains only the two final branch summaries.

First verify gradient flow and overfit a tiny two-hop synthetic batch. Then train
on the medical training split. Use the existing MedLatent diagnosis loss and
checkpoint conventions where convenient, but do not import the old centralized
all-hospital forward path as the distributed protocol.

### 11.4 Evaluation and non-goals

Evaluate only a loadable newly trained checkpoint on the same 401 N=10 test
episodes used by structured M2. Report diagnosis accuracy, mean latent-block
bytes/messages, `m`, and a trace proving B2 reaches the source through the
two-hop route. Compare directly with structured B2 and text B2.

Do not implement learned routing, GNN aggregation, topology search, dynamic
graphs, cross-family projection, policy gradients, communication penalties,
privacy adversaries, or a generalized latent registry in M3.

## 12. Verification Plan

### Unit invariants

- adjacency is symmetric, has no self-loops, and all sends obey `A`;
- per-agent fan-out never exceeds `k` in a round;
- messages arrive exactly one round after send;
- duplicate requests do not repeatedly activate an agent;
- no agent API exposes another agent's `PrivateKnowledgeStore`;
- transmitted byte accounting matches the payload representation.

### Integration tests

- a three-node path requires two hops to reach the evidence node;
- B0 fails and B2 succeeds on a constructed complementary-evidence example;
- reducing `R` or deleting the bridge edge removes that success;
- a distance-`d` evidence node cannot inform the source when `R < 2d` under
  request-return routing;
- the five-node ring produces B0/B1/B2 predictions and costs; and
- a fake text generator exercises the same route without loading a model.

Keep the three-node complementary-evidence fixture inline in a runtime or
medical test rather than maintaining a separate synthetic application. Real
Hugging Face inference stays outside the default fast test suite.

## 13. Milestones and Exit Criteria

### M0: freeze semantics and graph layer

Completed behavioral foundation: fixed undirected adjacency, legal-neighbor
sends, and shortest paths. Unused generators and serialization may be pruned.

### M1: structured distributed runtime

Completed behavioral foundation: private-store access, next-round delivery,
fan-out, request-return routing, and a constructed two-hop test. Unused config,
result-schema, and synthetic-baseline layers may be pruned.

### M2: medical and text baselines

Run the fixed five-hospital experiment in Section 8. Exit when B0/B1/B2
structured results exist for all 401 skewed test cases, the 50-case text pilot
runs on the same route, and per-case outputs show whether remote and two-hop
evidence help. Do not expand the experiment matrix before this result exists.

### M3: train a new fixed-route latent protocol

Build and train a new same-backbone latent communication interface on the fixed
`N=10` ring (`R=4`, `k=2`) protocol. Do not reuse old MedLatent checkpoints.
Exit only when a new distiller/boundary checkpoint is trained, a two-hop test
proves intermediate re-encoding, and the checkpoint is evaluated against the
M2 structured/text runs on the same episodes.

### M4: learned protocol

Learn aggregation, routing, and stopping one component at a time; add privacy
attacks and scaling to `N >= 100`. Exit when the method improves a documented
utility-cost-privacy frontier and the gain persists on tasks requiring
complementary evidence.

## 14. Open Decisions

The following should remain configurable or unresolved until experiments give
evidence:

1. Is the final target still rare-disease diagnosis, or is it only the first
   systems smoke before a stronger distributed-knowledge benchmark?
2. What exactly defines `I*(q)`, and can necessity be certified rather than
   approximated by retrieval rank or label occurrence?
3. Which agent metadata is public enough to support heuristic routing without
   weakening the private-knowledge claim?
4. Are agents honest-but-curious, can they collude, and can an observer see
   routing metadata as well as payloads?
5. Does the main method transmit token-like hidden states, layerwise KV blocks,
   or a model-agnostic transport latent?
6. Is communication budget enforced per edge, per sender, per episode, or all
   three in the final comparison?
7. When should directed graphs be introduced, and should their answer semantics
   require return-to-source or allow a designated terminal answer agent?

## 15. Task Tracker

This section is the implementation source of truth for future sessions. Work
roughly top-to-bottom and do not begin LLM integration before the synthetic
structured runtime is verified.

### 15.1 Update protocol for every session

Before coding, a session should read `RESEARCH_CONTEXT.md` and this file, then
select the earliest unblocked task. Use the following conventions:

- `[ ]`: not complete. Add `IN PROGRESS (YYYY-MM-DD, session note)` to the task
  line while actively working on it. A later session may reclaim it if no
  matching code/change is present.
- `[x]`: complete and acceptance criterion verified. Add `DONE (YYYY-MM-DD;
  evidence)` to the task line.
- Do not mark a task complete merely because code was written; its stated test
  or artifact must exist and pass.
- If blocked, keep `[ ]`, add `BLOCKED` plus the exact reason and the smallest
  decision/input needed.
- When scope or semantics change, update the relevant prose first, update the
  `Last updated` date at the top, then adjust affected tasks/dependencies.
- Preserve task IDs. Add new IDs instead of renumbering existing tasks so that
  experiment notes and commits can refer to stable identifiers.
- At the end of a session, run the narrow relevant tests, record the command
  and result on the task line, and leave unrelated checkboxes unchanged.

Current focus: `L3.1`. M2 is complete. The previous latent prototype was
deleted; implement the new KV-based trainable protocol below.

Completed planning tasks:

- [x] **D01 - Repository direction.** Extend the LatentDx
  `distributed-communication` branch under `medlatent/distributed/`; keep the
  current centralized path as a legacy/oracle baseline and use
  `DecentralizedMAS` as reference only. **DONE (2026-09-05; decision recorded
  in Section 7).**
- [x] **D02 - Phase 1 network scope.** Restrict the first implementation to
  fixed undirected graphs with symmetric adjacency, no self-loops, and
  synchronous next-round delivery. **DONE (2026-09-05; semantics recorded in
  Sections 2 and 4).**
- [x] **D03 - Reference inference backend.** Use Hugging Face `transformers`
  behind the `TextGenerator` protocol for reproducible text inference and later
  hidden/KV access; permit optimized interchangeable backends after validation.
  **DONE (2026-09-07; decision recorded in Section 8.1).**

### 15.2 Completed foundation

- [x] **FOUNDATION - M0/M1.** The fixed undirected graph, synchronous episode
  engine, private-store boundary, structured/text channels, B0-B4 code paths,
  event logging, and CPU smoke tests exist. **DONE (2026-09-07; commits
  `f1d394a` through `bd5d825`; detailed commands remain in the session log and
  `docs/distributed_m1_verification.md`).**

No new M0/M1 features are needed. M2L1 may delete unused foundation APIs while
preserving the tested graph, scheduler, and privacy-boundary behavior.

### 15.3 M2 - Lean baseline path

The old M207-M213 plan is retired. It expanded experiment bookkeeping before a
canonical result existed. Do not complete those tasks under their old scope.

- [x] **M2L0 - Behavioral reference.** Graph legality, next-round delivery,
  fan-out, private retrieval, B0/B1/B2 request-return, and TextMAS behavior have
  passing tests. **DONE (2026-09-07; commits through `bd5d825`).** These
  behaviors must survive pruning; the module/API structure need not.
- [x] **M2L1 - Prune to the Section 7.2 target.** Prune both committed and
  uncommitted framework code in two passes:

  **Delete whole vertical slices:** `config.py`, `results.py`, `fixtures.py`,
  the synthetic `baselines.py`, `sweeps.py`, `necessity.py`,
  `scripts/run_distributed_baseline.py`, `scripts/run_distributed_sweep.py`,
  `configs/distributed_baseline.yaml`, and their dedicated config/result/sweep/
  audit/CLI tests. Delete the uncommitted `medical_results.py` and its test.
  The necessary two-hop synthetic assertion moves into the episode or medical
  test rather than retaining a parallel baseline implementation.

  **Shrink retained modules:** keep only complete/ring/path and
  `neighbors`/`has_edge`/shortest-path/edge-list in `graph.py`; remove graph
  metadata/hash/save-load, star/random-regular, degree/component/diameter code.
  Remove ACK/EVIDENCE payloads, TTL/parent-message bookkeeping, and general
  message deserialization from `messages.py`; `R` already bounds propagation
  and the agent's parent node is enough for reply routing. Fold the tiny store
  protocol into `agent.py` or `medical.py`.
  Fold direct/flood routing and mean-score aggregation into
  `medical_baselines.py`, then delete `routing.py` and `aggregation.py`. Reduce
  `MedicalBaselineKind` to B0/B1/B2 and remove B3/B4, heuristic expertise,
  alternate aggregators, failure-stage taxonomy, empty-store counterfactuals,
  and all M207-M209 seed/mapping/oracle/statistics additions. Keep
  `agent.py`/`episode.py` scheduler semantics intact and make `__init__.py` a
  small export list.

  **Unify execution:** replace the synthetic/sweep/TextMAS scripts with one
  `scripts/run_distributed_medical.py`. It accepts data/model/output paths,
  channel, `max_samples`, and at most one seed; the canonical graph/`R`/`k` and
  methods live together in the script. Preserve only tests for graph symmetry,
  legal edge/fan-out, next-round delivery, two-hop return, private-store access,
  B0/B1/B2 predictions, cost accounting, and a fake text generator.

  **Done when:** the retained distributed package matches Section 7.2, one
  runner replaces the three old CLIs, the focused tests cover those invariants,
  and the full suite passes. Judge success by removed concepts and a readable
  end-to-end path, not by preserving the `bd5d825` APIs. **DONE (2026-09-07;
  `python -m pytest -q` in `DecentralizedMAS` passed 19 tests; one-case
  structured runner smoke passed).**
- [x] **M2L2 - Run the canonical structured baseline.** Use the fixed Section 8
  setting and existing code to run B0, B1, and B2 on all 401 skewed test cases.
  Emit the minimal per-case JSONL and per-method summary directly from the
  runner. Also run only the `R=2` ring and complete-graph reference. **Done
  when:** predictions, accuracy, messages, and bytes are saved and rerunnable
  from one documented command. **DONE (2026-09-08; artifacts under
  `outputs_distributed/`).** On all 401 skewed cases, ring `R=4`, `k=2` gave
  B0 `169/401` (42.14%), B1 `275/401` (68.58%), and B2 `303/401` (75.56%).
  Ring `R=2`, `k=2` left B2 at B1 accuracy; complete `R=2`, `k=4` matched ring
  B2 predictions exactly at 75.56%. Alias-aware evaluation did not change any
  structured count.
- [x] **M2L3 - Run the matched text pilot and decide.** Run B2 text on the same
  first 50 cases and fixed route/model, report source-local/remote/no-gold
  retrieval fractions using a direct retrieval scan, then apply the Section 8.4
  go/no-go criterion. **DONE (2026-09-08; `Qwen/Qwen3-4B-Instruct-2507`,
  ring `R=4`, `k=2`, first 50 canonical cases).** B2 text obtained `19/50`
  (38.0%) under declared-alias matching, compared with structured B2 `39/50`
  (78.0%) on the same cases, with 10 messages and 4308.02 mean bytes. Retrieval
  coverage was 38.0% source-local gold hit, 86.0% remote gold hit, and 10.0%
  no gold hit anywhere. **Decision: GO to M3.** Structured B2's full-set gain
  over B0/B1 is concentrated in reachable remote evidence; free-form text does
  not improve it and should remain a bandwidth/utility baseline.
- [x] **M2R1 - Ten-hospital structured robustness.** Run structured B0/B1/B2
  on all 401 cases for the `N=10` ring at `R=4,k=2`, the full-ring diagnostic
  at `R=10,k=2`, and the complete reference at `R=2,k=9`. **DONE (2026-09-08;
  official artifacts under `outputs_distributed/structured_10_*` reproduce B0
  `38.65%`, B1 `65.59%`, ring-R4 B2 `72.32%`, and full-reach B2 `73.57%`).**
- [ ] **M2R2 - Ten-hospital matched text pilot.** Run text B0/B1/B2 on the same
  first 50 cases, `N=10` ring, `R=4,k=2`, and
  `Qwen/Qwen3-4B-Instruct-2507`. Reuse the current runner/output schema and add
  no topology abstraction. **Done when:** artifacts are saved and one sentence
  selects `N=5` or `N=10` for M3 based on collaboration gain, text behavior,
  runtime, and communication cost.

### 15.4 M3 - New fixed-route latent protocol

- [x] **L3.1 - Define the KV protocol.** Implement the Section 11 contract:
  one incoming child summary (or stable ordered child summaries), one private
  top-1 local prompt, one newly re-encoded `m`-position block, and one final
  summary per source branch. Child summaries are aggregated at the relay and
  are not all returned separately. **Done
  when:** the code documents tensor/cache shapes and has no hidden-state or
  unrelated-checkpoint transport path. **DONE (2026-09-12; `distributed/latent.py`
  defines tuple KV blocks, stable child merge, and fresh relay re-encoding).**
- [x] **L3.2 - Differentiable KV rollout.** Freeze the backbone but initialize
  new distiller and boundary parameters. Preserve autograd through local
  decoder calls, latent rollout, KV slicing, and source prediction. **Done when:**
  a tiny two-hop batch gives nonzero gradients to both distiller and boundaries.
  **DONE (2026-09-12; `tests/test_distributed_latent.py`, focused and full suite
  pass in `DecentralizedMAS`).**
- [ ] **L3.3 - Train and checkpoint.** Train only on fixed N=10 ring/B2 episodes
  with diagnosis cross-entropy. Overfit a tiny synthetic batch first, then train
  on medical train data. Save a new loadable checkpoint with model ID, `m`, route
  settings, distiller, boundaries, and training metadata. **Done when:** loss
  decreases in the tiny smoke and a real checkpoint loads.
- [ ] **L3.4 - Protocol tests.** Test one-hop `R=2`, two-hop `R=4`, final-round
  delivery, parent relay, exactly one summary per branch, deterministic child /
  branch order, and no raw-record/text payload. **Done when:** tests fail for
  opaque forwarding, returning every child block separately, dropped final
  blocks, wrong shapes, or detached training tensors.
- [ ] **L3.5 - Matched evaluation.** Evaluate the new checkpoint on the same 401
  N=10 ring test episodes as M2 and compare accuracy and bytes with structured
  B2/text B2. Record checkpoint hash and command. **Done when:** one result
  table exists. No learned routing or alternate topology before this point.

### 15.5 Later work, not active tasks

GNN aggregation, learned routing, learned stopping, additional graph families,
larger `N`, heterogeneous backbones, formal privacy attacks, and robustness are
deferred. Promote only one of them into an active task when the M3 result shows
which limitation actually matters.

The active sequence is `L3.1 -> L3.2 -> L3.3 -> L3.4 -> L3.5`.

### 15.6 Session log

Append one concise row after any session that changes code, experiment
artifacts, task status, or settled research decisions. Commands should be
specific enough for the next session to rerun; use links to longer experiment
notes rather than expanding this table indefinitely.

| Date | Session/work | Tasks | Verification/evidence | Notes |
| --- | --- | --- | --- | --- |
| 2026-09-05/06 | M0/M1 and medical/text substrate | FOUNDATION | M1 verification note; distributed suite and CPU smoke passed | Foundation frozen |
| 2026-09-07 | Evidence-distance and remote-necessity fixes | M205, M206 | Commit `bd5d825`; 56 tests passed | Preserve the corrected concepts/tests where relevant; modules may be removed |
| 2026-09-07 | Generalized experiment layer | retired M207-M209 | Uncommitted working tree | Reviewed as excessive for the current question |
| 2026-09-07 | Lean scope reset | M2L1-M2L3 | Documentation only | Prune framework code across `bd5d825`, then run one fixed setting |
| 2026-09-07 | Deep foundation prune review | M2L1 | `bd5d825` dependency/line audit; no runtime code changed | Treat the commit as behavioral reference; target eight core modules and one runner |
| 2026-09-07 | Lean M2 runtime reduction | M2L1 | `DecentralizedMAS`: `python -m pytest -q` (19 passed); one-case structured `run_distributed_medical.py` smoke | Removed retired layers and APIs; canonical runner is ready for M2L2 |
| 2026-09-08 | GPU-runner preparation | M2L2, M2L3 | `DecentralizedMAS`: 19 tests passed; two real skewed cases passed for canonical, path, and complete settings; repeated structured artifacts were identical | Runner now records reproducibility metadata; [distributed M2 commands](docs/distributed_m2.md) are ready for the external instance |
| 2026-09-08 | Canonical M2 results and text pilot | M2L2, M2L3 | 401-case structured ring/ablation artifacts and 50-case Qwen text artifact under `outputs_distributed/` | B2 ring `R=4` improves B0/B1; text B2 is lower utility and higher bandwidth; GO to M3 |
| 2026-09-08 | Optional graph generators | deferred evaluation infrastructure | `DecentralizedMAS`: 21 tests passed, including seeded NetworkX generators | ER/Watts-Strogatz/SBM are importable through `medlatent[graphs]`, but deliberately not exposed by the fixed M2 runner |
| 2026-09-08 | Alias-aware distributed evaluation | M2L2, M2L3 | `DecentralizedMAS`: 22 tests passed; existing JSONL re-score left structured results unchanged and raised Qwen B2 first-50 from 16 to 19 correct | Accuracy now accepts declared `disease_aliases`; rerun artifacts to record the rule in their manifests |
| 2026-09-08 | Ten-hospital pre-M3 review | M2R1 | `DecentralizedMAS`: 22 tests passed; local 401-case structured diagnostics under `tmp/m2_review_10_*` | N=10 preserves the collaboration signal; run one official matched text pack, not a topology sweep |
| 2026-09-08 | Ten-hospital structured artifacts received | M2R1 | Three official artifacts under `outputs_distributed/structured_10_*`; values match the local diagnostic | Structured robustness complete; only matched N=10 text remains before M3 |
| 2026-09-08 | Ten-hospital text baseline and vLLM support | M2R2 | `outputs_distributed/text_10_ring_r4_k2`; commit `16fcca0` adds external vLLM endpoint support | M2 complete; next task L3.1 |
| 2026-09-12 | Rewrite M3 latent design after deleting prototype | L3.1-L3.5 | Documentation-only; new KV transport, fresh trainable checkpoint, fixed N=10 route | Next task: L3.1 |
| 2026-09-12 | Implement differentiable distributed KV protocol | L3.1, L3.2 | `DecentralizedMAS`: `python -m pytest -q` (25 passed) | Checkpointed frozen decoder; compact KV slices preserve gradients; relay re-encodes one `(m+2)` block |
