"""
SynapseOS World Model.

The integration layer that turns disconnected models + a database into a
persistent cognitive substrate. Built on the principle from JEPA:

    The world model predicts the next latent. What it FAILS to predict
    (prediction error = surprise) is what becomes a memory.

Components:
  encoder.py            — perception: text → observation embedding o_t
  world_model.py        — WorldModelNet (CfC state evolution + predictor) + WorldModel orchestrator
  state.py              — WorldState: the persistent, serialized, versioned z_t
  predictive_memory.py  — surprise-gated episodic write with temporal/causal links
  retrieval.py          — state-conditioned associative recall in latent space
  consolidation.py      — replay high-salience episodes, distill to semantic, reduce future surprise

Everything runs on CPU/MPS, no GPU required.
"""

from synapseos.world_model.world_model import WorldModel, PerceptionResult
from synapseos.world_model.state import WorldState

__all__ = ["WorldModel", "PerceptionResult", "WorldState"]
