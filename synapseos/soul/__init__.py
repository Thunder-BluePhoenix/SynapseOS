"""
SynapseOS Soul — the conditions for something soul-like to grow.

You can't build a soul; you build the substrate and live with it (per the
companion vision docs). Continuity (world model), emotional valence (affect),
and beliefs (web belief reviser) already exist. This package adds the three
that were missing:

  aesthetics  — taste. Records reactions to things encountered (a proof, a
                melody, an analogy) and develops STABLE preferences from
                exposure. The companion can say "I find this beautiful, and
                here's why" — grounded in accumulated history, not flattery.
  reflection  — growth it can see. Periodic snapshots of its own state;
                compares now against then and narrates what changed:
                "a month ago I leaned X; after Y, I now think Z."
  values      — moral courage. Internalized positions that update on argument
                but NOT on pressure. It can say "I could, and I won't" and
                pay the cost — distinct from external safety rules.
"""

from synapseos.soul.aesthetics import AestheticPreferences, Reaction
from synapseos.soul.reflection import SelfReflection, Snapshot
from synapseos.soul.values import ValueSystem, ValueStance, MoralJudgment

__all__ = ["AestheticPreferences", "Reaction", "SelfReflection", "Snapshot",
           "ValueSystem", "ValueStance", "MoralJudgment"]
