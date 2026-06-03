# Phase 4 — Advanced Capabilities
**Timeline:** Weeks 13–20

---

## Goal

Complete all 5 thinking modes. Add multimodal inputs. Run RL fine-tuning to align the full system. This phase turns NEXUS into a capable cognitive system, not just a connected pipeline.

---

## Tasks

### Creativity Engine (Mode 2 + Mode 3)

- [ ] Implement knowledge graph random walk (for cross-domain exploration)
- [ ] Build analogy engine — detect structurally similar but domain-distant nodes
  - Example: "ant colony routing" + "network latency" → novel routing algorithm idea
- [ ] Implement evolutionary mutation: take existing idea, apply valid transformation, score result
- [ ] Build multi-agent debate loop: spawn internal reasoning agents, synthesize disagreements
- [ ] Add confidence + novelty scorer: rank generated ideas before surfacing
- [ ] Test Mode 3 (New Thought Synthesis): verify each new idea has a traceable generation path

```
1. Random walk → find structurally similar but distant nodes
2. Analogy engine → cross-domain bridging
3. Evolutionary mutation → apply valid transformation
4. Multi-agent debate → internal disagreement → emergent synthesis
5. Constraint forcing → "solve X under constraint Y"
6. Confidence scorer → rank by plausibility × novelty
```

### Program Synthesis Module (Mode 4 — Code Thinking)

- [ ] Build intent parser: extract function goal, inputs, constraints from natural language
- [ ] Connect to API/code knowledge graph: what functions, classes, patterns exist
- [ ] Implement algorithm planner: decompose goal into symbolic sub-steps
- [ ] Build program synthesizer: construct code from symbolic plan (not token prediction)
- [ ] Integrate static analyzer: verify code correctness before output
- [ ] Add test auto-generator: produce verification cases for every synthesized function

```
1. Parse intent → goal, inputs, constraints
2. Query knowledge graph → available APIs, patterns
3. Plan algorithm → decompose to symbolic steps
4. Synthesize code from plan
5. Static analyze → verify before output
6. Auto-generate tests
```

### Idle Exploration Loop (Mode 5 — Default Mode Network)

- [ ] Implement idle trigger: detect N seconds with no active task
- [ ] Build background graph walk: explore low-confidence edges during idle time
- [ ] Pattern consolidation: compress recent episodic memories into semantic facts
- [ ] Weak hypothesis store: save low-confidence ideas for future reinforcement
- [ ] Wire liquid layer background update: adapt weights from recent interaction patterns
- [ ] Implement wake signal: resume from idle state without losing accumulated context

### Multimodal Input — Vision

- [ ] Integrate CLIP encoder for image inputs
- [ ] Connect CLIP output to JEPA fusion layer (same latent space as text/code)
- [ ] Test: image + text joint reasoning queries

### Multimodal Input — Audio

- [ ] Integrate Whisper encoder for audio/voice inputs
- [ ] Connect Whisper transcription output to Mamba encoder
- [ ] Test: voice query → full cognitive pipeline → natural language output

### RL Fine-Tuning — End-to-End Alignment

- [ ] Define reward function (human ratings, automated quality metrics, or both)
- [ ] Run policy gradient training end-to-end (unfreeze top layers, keep bottom layers frozen)
- [ ] GPT-4 distillation: generate (input, output) pairs from GPT-4, train student to match on domain tasks
- [ ] Evaluate alignment: outputs should be more accurate, less verbose, more faithful to reasoning

---

## Thinking Modes Coverage After This Phase

| Mode | Name | Status |
|---|---|---|
| Mode 1 | Focused Reasoning | Complete in Phase 3 |
| Mode 2 | Creative Generation | Complete this phase |
| Mode 3 | New Thought Synthesis | Complete this phase |
| Mode 4 | Code Thinking | Complete this phase |
| Mode 5 | Idle Exploration | Complete this phase |

---

## Inputs

| Input | Source |
|---|---|
| Integrated NEXUS pipeline | Phase 3 output |
| Knowledge graph | Phase 1 output |
| CLIP pretrained weights | HuggingFace / OpenAI |
| Whisper pretrained weights | HuggingFace / OpenAI |
| Human ratings or quality signal | Manual annotation or automated metrics |
| GPT-4 teacher outputs | API |

## Outputs

| Output | Format |
|---|---|
| Creativity engine module | Python module |
| Program synthesis module | Python module |
| Idle exploration loop | Background process / daemon |
| Multimodal-capable pipeline | Updated inference stack |
| RL-aligned system weights | Saved checkpoints |

---

## Compute Estimate

| Component | GPU | Time | Cost |
|---|---|---|---|
| Creativity + synthesis modules | CPU-heavy | 1–2 weeks dev | Minimal |
| Multimodal integration (CLIP + Whisper) | 1× A100 | 4–8 hrs | $50–150 |
| RL fine-tuning | 2× A100 | 12–24 hrs | $100–500 |
| GPT-4 distillation data | API | — | $30–100 |

---

## Success Criteria

- [ ] All 5 thinking modes execute end-to-end on test inputs
- [ ] Creativity engine produces ideas with traceable generation paths (not opaque token outputs)
- [ ] Program synthesis produces syntactically valid, statically verified code on 90%+ of test cases
- [ ] Idle loop runs without degrading active-task performance
- [ ] Voice input → output pipeline works end-to-end
- [ ] RL-tuned model scores higher on quality metrics than Phase 3 baseline

---

## Dependencies

- [Phase 3 — Integration](Phase3_Integration.md) complete (full pipeline wired, Modes 1 and 4 baseline working)

## Unlocks

[Phase 5 — SynapseOS](Phase5_SynapseOS.md)
