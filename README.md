# SynapseOS

**A cognitive architecture that is not an LLM wrapper.**

SynapseOS is an experiment in building a "nervous-system-shaped" AI: knowledge lives in an explicit, auditable graph; reasoning is symbolic and rule-checked; memory persists across sessions; and the system keeps learning, feeling, wondering and creating between conversations. Text generation is the *last* step, not the first.

> **Status: research prototype.** It runs end-to-end on a laptop (CPU / Apple Silicon MPS) and has a large test suite, but the heavy neural components (Mamba-7B fine-tuning, output-synthesizer SFT, RL) are not trained at full scale. See [Honest status](#honest-status).

---

## Why

LLMs compress knowledge into weights, recompute everything per request, forget between sessions, and cannot verify their own output. SynapseOS takes the opposite bet:

| | LLM | SynapseOS |
|---|---|---|
| Knowledge | Compressed into weights | Explicit knowledge graph + rules |
| Reasoning | Statistical pattern matching | Symbolic, checked against known rules |
| Memory | Stateless per session | Episodic + semantic + working, persisted |
| Novel ideas | Opaque token recombination | Graph walks / analogies with a traceable generation path |
| Interpretability | Black box | KAN layers distil to readable formulas |
| Adaptation | Retraining | Liquid layers + a world model that updates online |

The full design, with citations, is in [`docs/SynapseOS_Architecture.md`](docs/SynapseOS_Architecture.md).

## Architecture

The design (NEXUS → SynapseOS) layers five model families, each with one job:

```
input ─► Mamba (sequence) ─► JEPA (world state) ─► Neurosymbolic core (KG · rules · validator)
                                                        │
                       Liquid NN (adapts online) ◄──────┤
                       KAN (interpretable math)  ◄──────┘
                                                        ▼
                                              Output synthesizer
```

On top of that sits `CognitiveController` (`synapseos/phase5/cognitive_controller.py`), the single entry point. Each `think(query)` call:

1. Pulls memory context (episodic, semantic, working).
2. Reads the emotion in the message and updates a persistent emotional state.
3. Checks the request against held **values**. A violation short-circuits into a refusal that names the value.
4. Lets the **world model** perceive the query, which yields surprise and novelty, then recalls related past experience.
5. Routes to **thinking modes** and **agents** in parallel threads.
6. Synthesizes the result, adds recalled experience, and modulates tone to the user's affect.
7. Records the interaction, and lets the world model perceive the system's own response.

### Thinking modes

| Mode | Purpose | Implementation |
|---|---|---|
| 1 Focused reasoning | Verified answers from the KG and rules | `phase3/pipeline.py`, `neurosymbolic_core.py` |
| 2 Creative generation | Structured ideas | `phase4/creativity_engine.py` |
| 3 New-thought synthesis | Cross-domain analogy, mutation, multi-agent debate | `phase4/creativity_engine.py` |
| 4 Code thinking | Intent → plan → synthesized, statically-checked code | `phase4/program_synthesis.py` |
| 5 Idle exploration | Background graph walks and memory consolidation | `phase4/idle_loop.py` |

### Agents

Thirteen specialist agents, scored per query and dispatched in parallel (`synapseos/phase5/agents/`):
**ARIEL** (business logic), **NYX** (security), **NOVA** (optimization), **ATLAS** (infrastructure), **KARMA** (actions), **MUSE** (creative output), **MENTOR** (teaching, exact math), **VITA** (health), **LEX** (legal), **AURUM** (finance), **FORGE** (engineering), **SAGE** (counselling), **ORACLE** (foresight).

### Subsystems

| Package | What it does |
|---|---|
| `world_model/` | Persistent latent state *z_t*; **prediction error decides what becomes a memory**; state-conditioned recall; consolidation into semantic memory |
| `affect/` | Rule-based emotion detection from text and speech prosody; tracked emotional state; response modulation (no neural language model) |
| `soul/` | Aesthetic taste, self-reflection snapshots, values that update on argument but not on pressure, and a "wonder" journal of open questions |
| `web/` | Curiosity-driven web learning: polite fetcher, fact extraction, trust-weighted belief revision, curriculum, YouTube/audio ingestion |
| `actions/`, `identity/` | Files, sandboxed shell, HTTP, browser, macOS control, scheduler. Every action is permission-gated and audit-logged. Outbound messages wait in an approval **outbox** |
| `gateway/` | Telegram and Discord presence; every inbound message goes through `think()`, with a chat allowlist and audit trail |
| `mesh/` | Offline-first, authenticated peer sync of memory, wonders, affect, taste and values across devices ([details](#device-mesh)) |
| `create/` | Deterministic, seeded music (incl. Hindustani raags), songs, lyrics, images, posters, video, workflows |
| `skills/` | Exact math solver (arithmetic to calculus and statistics) and language detection |
| `local/` | Install-free CPU/MPS stand-ins for GPU-only parts: pure-PyTorch Mamba scan, CfC liquid cell, SFT trainer, whisper.cpp STT, macOS `say` TTS, SigLIP/CLIP vision, and an I-JEPA trained from scratch |
| `api/`, `eval/` | FastAPI server and a quality evaluator |

## Quickstart

Requires Python 3.11+ and PyTorch. Developed on Apple Silicon (macOS); core features run on CPU anywhere. Voice and TTS use macOS-specific tools.

```bash
git clone <this-repo> && cd SynapseOS
python -m venv .venv && source .venv/bin/activate
pip install torch numpy networkx scipy scikit-learn tqdm fastapi uvicorn pydantic
```

`requirements.txt` also lists CUDA-only packages (`mamba-ssm`, `ncps`, `trl`, ...) for the full-scale GPU path. You do **not** need them to run the local system.

The repo ships a prebuilt knowledge graph (1,385 nodes, 3,101 edges, 1,106 DocTypes), 2,414 rules, and small trained checkpoints under `data/`, so it works out of the box.

### Ask it something

```bash
python -m synapseos.phase5.cognitive_controller --query "What fields are required for a Sales Invoice?"
```

```python
from synapseos.phase5.cognitive_controller import CognitiveController

c = CognitiveController()
r = c.think("What validate rules apply to Sales Invoice?")
print(r.primary_output, r.confidence, r.routed_to_agent, r.surprise)
c.shutdown()   # persists memory, world model, emotional state
```

### More entry points

```bash
python scripts/demo_synapseos.py                  # tour: modes, agents, creativity, voice, drift, pipeline, memory
python scripts/demo_web_learning.py --offline     # web learning loop on canned fixtures, no network
python scripts/benchmark_pipeline.py              # concurrent load test (p50 / p99 / throughput)
python -m synapseos.api.server --port 8765        # REST API; docs at http://localhost:8765/docs
python -m synapseos.local.device                  # which accelerator is selected (MPS > CUDA > CPU)
```

REST endpoints: `/query`, `/query/mode/{mode}`, `/query/agent/{name}`, `/agents`, `/stats`, `/memory/*`, `/kg/*`, `/voice/process`, `/benchmark`.

### Local training demos

```bash
python -m synapseos.local.mamba_cpu    --train-demo   # selective-scan SSM
python -m synapseos.local.liquid_cfc   --train-demo   # closed-form liquid cell
python -m synapseos.local.sft_trainer  --demo         # tiny decoder SFT
python -m synapseos.local.jepa_vision  --train --steps 2500   # I-JEPA on CIFAR-10 (downloads data)
python -m synapseos.world_model.world_model           # world model: learn a routine, flag novelty, recall
```

## Device mesh

`synapseos/mesh/` keeps several devices in sync without a cloud: last-write-wins oplog, offline queue, deterministic merge.

It syncs facts (deletions as tombstones; incoming facts pass through the belief reviser), wonders, episode summaries, emotional state, taste, values and reflection snapshots.

**Security is mandatory.** Every request and response is HMAC-SHA256 signed with a shared key, checked against a peer allowlist, and protected against replay and stale timestamps. HMAC does not encrypt, so a non-loopback listener or peer **requires TLS**. There is no unauthenticated mode.

```python
c = CognitiveController(
    enable_mesh=True,
    mesh_shared_key="<32+ random bytes, same on every device>",
    mesh_allowed_peer_ids=["laptop", "phone"],      # node ids allowed to talk to this device
    mesh_peers=["https://phone.local:7700"],
    mesh_server_host="0.0.0.0",                     # optional listener; needs mesh_ssl_context off-loopback
    mesh_ssl_context=my_ssl_context,
)
```

Sync starts automatically. With no key and allowlist (or an explicit transport) the mesh stays disabled and `c.mesh_error` says why.

Caveat: one shared key means any key-holder can claim any allowlisted node id. Use a separate key per mesh and rotate it if a device is lost.

## Safety model

- **Actions** pass a `PermissionPolicy` (trust level 0-3, workspace sandbox, command allowlist, dry-run) *before* running, and land in `data/actions/audit.json`.
- **Outbound email and posts** are drafted into the **outbox** and sent only after approval.
- **Gateway bots** announce that they are AI, answer only allowlisted chats, and log every message.
- **Credentials** come from environment variables or `data/identity/credentials.json` (gitignored). Nothing is hardcoded.

Run the companion under a user account and workspace you are comfortable with. The action layer can execute commands.

## Tests

```bash
python tests/test_mesh.py      # most test files are self-running scripts with their own runner
python tests/test_phase5.py
python -m pytest tests -q      # runs the pytest-style tests; custom-runner files report a collection error for their `test` helper
```

Test files such as `test_mesh.py`, `test_soul.py` and `test_phase5.py` use their own `@test` runner, so run them as scripts. Under `pytest` each shows a harmless collection error for the `test` decorator, while the remaining tests pass. Some voice tests depend on `ffmpeg` and whisper.cpp being installed.

## Honest status

| Area | State |
|---|---|
| KG, rules, neurosymbolic pipeline, 5 modes, 13 agents | Working, tested |
| World model, affect, soul, web learning, actions, gateway, mesh, create | Working, tested |
| JEPA / KAN / Liquid / bridges | Trained at small scale on CPU (synthetic or modest data) |
| Local Mamba, CfC, SFT, I-JEPA | Real but small stand-ins, not equivalent to the full-scale models |
| Mamba-7B LoRA, output-synthesizer SFT (Mistral/Llama), RL fine-tuning | **Not done.** Needs a CUDA GPU (estimated $545-$2,080) |
| Live Liquid ODE adaptation (`ncps`) | Not done (needs the CUDA path) |

Claims such as "near-zero hallucination" and "fully auditable" are **design goals**. Much of the current reasoning is keyword routing and deterministic rules over a graph extracted from the Frappe/ERPNext codebase, and its answers are only as good as that graph. The KG extraction scripts (`scripts/extract_graph.py`, `scripts/extract_rules.py`) point at a local Frappe bench path and must be edited to run on your machine.

Progress by phase lives in [`docs/TRACKER.md`](docs/TRACKER.md).

## Repository layout

```
synapseos/   source package (phase1-5, world_model, affect, soul, web, actions, identity,
             gateway, mesh, create, skills, local, api, eval)
scripts/     demos, benchmarks, KG/corpus extraction
tests/       test suites
docs/        architecture, per-phase specs, tracker
data/        knowledge graph, rules, checkpoints, and runtime state (mostly gitignored)
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
