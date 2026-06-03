# Phase 2 — Neural Components
**Timeline:** Weeks 5–8

---

## Goal

Train each neural layer independently in parallel. All four components can train simultaneously — do not serialize unless constrained by GPU availability.

---

## Tasks

### Mamba Sequence Engine — Fine-Tuning

- [ ] Launch full LoRA fine-tuning on Falcon Mamba 7B with complete domain corpus
- [ ] Monitor training loss and domain perplexity throughout
- [ ] Save checkpoints every 2 hours; keep best 3 by validation perplexity
- [ ] Evaluate: run inference on held-out domain samples, compare to base model

```python
from mamba_ssm import MambaLMHeadModel
from peft import LoraConfig, get_peft_model

model = MambaLMHeadModel.from_pretrained("tiiuae/falcon-mamba-7b")
lora_cfg = LoraConfig(r=16, target_modules=["x_proj", "dt_proj"])
model = get_peft_model(model, lora_cfg)
# loss: causal LM (next token prediction on domain corpus)
```

### JEPA World State Model — Training

- [ ] Prepare state transition sequences from workflow histories and event logs (~50k sequences)
- [ ] Configure I-JEPA with context/target encoder architecture
- [ ] Train with L2 loss in latent space (no token prediction)
- [ ] Validate: predicted latent states should cluster by semantic similarity

```python
import torch.nn.functional as F
# loss: L2 in latent space — not token prediction
loss = F.mse_loss(predicted_latent, target_latent.detach())
```

### KAN Reasoning Layer — Training

- [ ] Collect domain metric pairs (e.g., [approver_count, amount, complexity, deadline] → approval_time)
- [ ] Train KAN with grid=5, k=3 on collected pairs
- [ ] Run `auto_symbolic()` to extract readable formula after training
- [ ] Document extracted formulas — these become auditable business rules

```python
from kan import KAN
model = KAN(width=[4, 8, 1], grid=5, k=3)
model.auto_symbolic()
# → "outcome = 8.3 / factor_1^1.8 + 0.4 * factor_2"
```

### Liquid Neural Network Layer — Training

- [ ] Collect time-series data: CPU, memory, queue depths, DB latency, APM metrics, event streams
- [ ] Configure LTC with AutoNCP wiring (32 neurons, 8 outputs as baseline)
- [ ] Train on next-state prediction objective
- [ ] Confirm ODE-based adaptation works at inference: weights should shift as new inputs arrive

```python
from torchdiffeq import odeint_adjoint
from ncps import wirings, LTC

wiring = wirings.AutoNCP(32, 8)
ltc = LTC(input_size, wiring)
# weights continue adapting at inference via ODE solver
```

---

## Inputs

| Input | Source |
|---|---|
| Domain corpus (tokenized) | Phase 1 output |
| Event logs / workflow histories | Production system or synthetic |
| Domain metric pairs | Analytics DB / operational data |
| Time-series monitoring data | APM, infra metrics |

## Outputs

| Output | Format |
|---|---|
| Mamba fine-tuned weights | Saved checkpoint + LoRA adapter |
| JEPA world state encoder | Saved PyTorch model |
| KAN trained model + symbolic formulas | `.pt` + extracted formula text |
| Liquid NN trained model | Saved checkpoint |

---

## Compute Estimate

| Component | GPU | Time | Cost |
|---|---|---|---|
| Mamba fine-tuning | 2× A100 80GB | 12–48 hrs | $200–800 |
| JEPA world model | 1× A100 40GB | 6–24 hrs | $50–200 |
| Liquid NN | 1× T4 | 2–6 hrs | $10–30 |
| KAN layer | CPU or 1× T4 | 1–4 hrs | $5–20 |

All four can run in parallel if GPUs are available.

---

## Success Criteria

- [ ] Mamba perplexity on domain holdout better than base model baseline
- [ ] JEPA latent space clusters align with semantic categories (verify with t-SNE)
- [ ] KAN produces at least one readable symbolic formula for a known relationship
- [ ] Liquid NN: inference weights measurably shift between different input regimes

---

## Dependencies

- [Phase 1 — Foundation](Phase1_Foundation.md) complete (knowledge graph built, corpus ready)

## Unlocks

[Phase 3 — Integration](Phase3_Integration.md)
