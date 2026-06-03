# Phase 1 — Foundation
**Timeline:** Weeks 1–4

---

## Goal

Build the knowledge layer and sequence modeling backbone. Everything downstream depends on these two being solid.

---

## Tasks

### Knowledge Graph Extraction

- [ ] Write AST parser to extract entities, fields, relationships, and workflows from domain codebase
- [ ] Serialize extracted data to graph DB (Neo4j or NetworkX)
- [ ] Write domain-specific business rules in Datalog (IF condition THEN consequence triples)
- [ ] Validate graph coverage — every major entity and relationship should be represented

### Mamba Sequence Engine Setup

- [ ] Install dependencies: `mamba-ssm`, `peft`, `transformers`
- [ ] Source and clean domain corpus (~500M tokens — source code, documentation, domain text)
- [ ] Validate Falcon Mamba 7B loads correctly and runs inference
- [ ] Run baseline LoRA fine-tuning test (small subset) to confirm pipeline is functional

---

## Inputs

| Input | Source |
|---|---|
| Domain codebase | Local repo |
| Domain documentation | Internal docs, wikis |
| Business rules | Domain experts / existing rule files |

## Outputs

| Output | Format |
|---|---|
| Knowledge graph | Neo4j DB or `graph.json` (NetworkX) |
| Datalog rules file | `.dl` or `.py` |
| Mamba fine-tuning pipeline | Runnable training script |
| Domain corpus | Tokenized dataset |

---

## Tech Stack

| Component | Library/Tool |
|---|---|
| AST parsing | Python `ast`, `tree-sitter` |
| Graph DB | Neo4j or NetworkX |
| Symbolic rules | Datalog (Souffle or PyDatalog) |
| Sequence model | `mamba-ssm` (Falcon Mamba 7B) |
| Fine-tuning | `peft` (LoRA), `transformers` |

---

## Compute Estimate

| Task | GPU | Time | Cost |
|---|---|---|---|
| Knowledge graph build | CPU only | 1–2 weeks | $0 |
| Mamba LoRA fine-tuning | 2× A100 80GB | 12–48 hrs | $200–800 |

---

## Success Criteria

- [ ] Knowledge graph populated — entities, relationships, rules queryable
- [ ] Mamba fine-tuning pipeline runs end-to-end without errors
- [ ] Baseline domain perplexity measured and documented
- [ ] Graph coverage report generated

---

## Dependencies

None — this is the base phase.

## Unlocks

[Phase 2 — Neural Components](Phase2_Neural_Components.md)
