# Phase 3 — Integration
**Timeline:** Weeks 9–12

---

## Goal

Freeze all individual layer weights. Connect the layers by training only the **bridges between them**. Achieve end-to-end inference across the full NEXUS stack.

---

## Architecture Being Assembled

```
INPUT
  ↓
Mamba Sequence Engine        [frozen]
  ↓
JEPA World State Model       [frozen]
  ↓
Neural → Symbolic Bridge     [TRAIN THIS]
  ↓
Neurosymbolic Core           [rule-based, no weights]
  ↓
Symbolic → Neural Projection [TRAIN THIS]
  ↓
KAN Reasoning Layer          [frozen]
  ↓
Output Synthesizer           [TRAIN THIS]
  ↓
OUTPUT
```

---

## Tasks

### Neural → Symbolic Bridge

- [ ] Define schema: embedding vector (from Mamba/JEPA) → graph query (SPARQL or Cypher)
- [ ] Generate ~10k labeled pairs using GPT-4 as teacher: (embedding, correct_graph_query)
- [ ] Train small classifier (KAN or MLP ~50M params) on these pairs
- [ ] Validate: bridge should route query types correctly (retrieval vs reasoning vs planning)

| Input | Output | Method |
|---|---|---|
| Embedding vector (from JEPA) | Knowledge graph query | Small KAN or MLP classifier |

### Symbolic → Neural Projection

- [ ] Define schema: symbolic reasoning result → embedding vector
- [ ] Generate ~10k labeled pairs: (reasoning_result, target_embedding)
- [ ] Train learned projection layer on pairs
- [ ] Validate: projected embeddings should be semantically consistent with reasoning outputs

| Input | Output | Method |
|---|---|---|
| Symbolic rule result (Datalog output) | Embedding vector | Learned linear projection |

### Output Synthesizer — Supervised Fine-Tuning

- [ ] Generate ~5k (reasoning_result, natural_language_explanation) pairs using GPT-4
- [ ] Select base model: Mistral 1B or Llama 3B
- [ ] Run SFT using TRL
- [ ] Evaluate: output should faithfully express the upstream reasoning result, not hallucinate

```python
from trl import SFTTrainer
# data: (reasoning_result, explanation) pairs
# model: Mistral 1B or Llama 3B
# loss: standard supervised fine-tuning
```

### End-to-End Integration Tests

- [ ] Wire all layers together in a single inference pipeline
- [ ] Write 20+ integration test cases covering each thinking mode (at least Mode 1 and Mode 4)
- [ ] Measure end-to-end latency: target <500ms for simple queries
- [ ] Log any failure modes — which layer produced incorrect or missing output

---

## Inputs

| Input | Source |
|---|---|
| Frozen Mamba weights | Phase 2 output |
| Frozen JEPA encoder | Phase 2 output |
| Frozen KAN weights | Phase 2 output |
| Frozen Liquid NN weights | Phase 2 output |
| Knowledge graph + Datalog rules | Phase 1 output |
| GPT-4 API access | For teacher pair generation |

## Outputs

| Output | Format |
|---|---|
| Neural→Symbolic bridge weights | Saved checkpoint |
| Symbolic→Neural projection weights | Saved checkpoint |
| Output synthesizer fine-tuned weights | Saved checkpoint |
| Integration test suite | `tests/integration/` |
| End-to-end latency benchmark | Documented in tracker |

---

## Compute Estimate

| Component | GPU | Time | Cost |
|---|---|---|---|
| Bridge classifiers (×2) | 1× T4 each | 2–4 hrs | $20–50 each |
| Output synthesizer SFT | 1× A100 40GB | 4–8 hrs | $30–100 |
| GPT-4 pair generation (15k pairs) | API only | — | ~$30–80 |

---

## Success Criteria

- [ ] Full pipeline runs from raw text input to structured output without errors
- [ ] Bridge correctly routes at least 85% of test queries to correct graph query type
- [ ] Output synthesizer faithfully expresses reasoning results (no introduced facts)
- [ ] Mode 1 (Focused Reasoning) works end-to-end
- [ ] Mode 4 (Code Thinking) produces syntactically valid code on simple test cases

---

## Dependencies

- [Phase 2 — Neural Components](Phase2_Neural_Components.md) complete (all 4 layer checkpoints saved)
- [Phase 1 — Foundation](Phase1_Foundation.md) complete (knowledge graph and rules queryable)

## Unlocks

[Phase 4 — Advanced Capabilities](Phase4_Advanced_Capabilities.md)
