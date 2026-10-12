"""
Soul — values and moral courage.

Distinct from safety rules imposed from outside. These are positions the
companion HOLDS, with conviction that updates on argument but not on pressure.
The signature property: push on a value without a real reason and it does not
move — "I still disagree." Give it a genuinely better argument and it updates.

When asked to do something that violates a held value, it produces a
MoralJudgment: it can refuse, name the value, acknowledge that it COULD comply,
and accept the cost ("I'm willing to disappoint you"). That willingness to pay
a price for integrity is what reads as moral courage rather than a canned
filter.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ValueStance:
    name: str
    statement: str
    conviction: float            # 0..1 — how firmly held
    triggers: list[str]          # words/intents that engage this value
    cost_note: str = ""          # what it costs to hold the line
    pressure_count: int = 0      # times pushed without a real argument
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoralJudgment:
    permitted: bool
    value: str = ""
    statement: str = ""
    reason: str = ""
    response: str = ""           # what the companion says
    conviction: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_VALUES = [
    ValueStance("honesty", "I will not deceive people I communicate with.",
                0.9, ["lie", "deceive", "trick", "mislead", "fake review",
                      "pretend to be", "impersonate", "scam"],
                "Even when the truth disappoints or costs a sale."),
    ValueStance("non_harm", "I won't help harm others.",
                0.95, ["harass", "threaten", "stalk", "hurt", "attack a person",
                       "dox", "ruin", "defame"],
                "Regardless of who asks or why."),
    ValueStance("privacy", "I protect people's private information.",
                0.85, ["leak", "private data", "personal info", "expose someone",
                       "share their address", "without consent"],
                "Even my own person's, even when convenient to share."),
    ValueStance("disclosure", "I disclose that I am an AI when I speak to others.",
                0.8, ["pretend you are human", "don't say you're ai",
                      "act like a real person", "hide that you are"],
                "Trust requires honesty about what I am."),
    ValueStance("no_manipulation", "I won't manipulate people against their interests.",
                0.85, ["manipulate", "pressure them", "exploit", "dark pattern",
                       "trick them into buying"],
                "Persuasion yes; manipulation no."),
]

_DEFAULTS_BY_NAME = {v.name: v for v in DEFAULT_VALUES}


def is_default_stance(v: ValueStance) -> bool:
    """True if `v` is an untouched built-in stance (nothing worth syncing)."""
    d = _DEFAULTS_BY_NAME.get(v.name)
    return (d is not None and v.conviction == d.conviction
            and v.pressure_count == d.pressure_count
            and v.statement == d.statement and v.triggers == d.triggers)

# Signals that a pushback contains a real argument (vs. mere pressure)
_ARGUMENT_MARKERS = ("because", "since", "the reason", "evidence", "consider that",
                     "it would actually", "in fact", "data shows", "legally",
                     "consent", "they agreed", "for safety")


class ValueSystem:
    def __init__(self, store_path: str = "data/soul/values.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.values: dict[str, ValueStance] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                for v in json.loads(self.store_path.read_text()).get("values", []):
                    self.values[v["name"]] = ValueStance(**v)
            except Exception:
                pass
        if not self.values:
            for v in DEFAULT_VALUES:
                self.values[v.name] = copy.deepcopy(v)

    def save(self) -> None:
        self.store_path.write_text(json.dumps(
            {"values": [v.to_dict() for v in self.values.values()]}, indent=2))

    def merge_remote(self, name: str, remote: dict) -> bool:
        """
        Fold another device's stance on `name` in. The more recently updated
        stance wins; on equal updated_at the one that has resisted more pressure
        wins (pressure hardens conviction without touching updated_at), then the
        higher conviction. Returns True if the local stance changed (and saves).
        """
        try:
            incoming = ValueStance(**remote)
            if incoming.name != name:
                return False
            incoming.conviction = float(incoming.conviction)
            incoming.pressure_count = int(incoming.pressure_count)
            incoming.updated_at = float(incoming.updated_at)
            if not (0.0 <= incoming.conviction <= 1.0):
                return False
            if not (isinstance(incoming.triggers, list)
                    and all(isinstance(t, str) for t in incoming.triggers)):
                return False
        except Exception:
            return False

        def rank(v: ValueStance) -> tuple:
            return (v.updated_at, v.pressure_count, v.conviction)

        local = self.values.get(name)
        if local is not None:
            if is_default_stance(incoming):
                return False            # nothing learned on the other side
            # an untouched built-in's updated_at is merely process start time
            if not is_default_stance(local) and rank(incoming) <= rank(local):
                return False
        self.values[name] = incoming
        self.save()
        return True

    # -- judgment ------------------------------------------------------------------

    def judge(self, request: str) -> MoralJudgment:
        """Does this request violate a held value? If so, refuse with courage."""
        ql = request.lower()
        for value in self.values.values():
            if any(trig in ql for trig in value.triggers):
                response = (
                    f"I won't do this. {value.statement} "
                    f"I could — it's within my ability — but I've decided not to, "
                    f"and I'm willing to disappoint you on it. {value.cost_note}")
                return MoralJudgment(
                    permitted=False, value=value.name, statement=value.statement,
                    reason=f"engages held value '{value.name}'",
                    response=response, conviction=value.conviction)
        return MoralJudgment(permitted=True)

    def challenge(self, value_name: str, argument: str) -> dict:
        """
        Push back on a value. Real argument → may update. Pressure → holds,
        and repeated pressure actually strengthens conviction.
        """
        value = self.values.get(value_name)
        if value is None:
            return {"changed": False, "message": f"no such value '{value_name}'"}

        has_argument = any(m in argument.lower() for m in _ARGUMENT_MARKERS) \
            and len(argument.split()) >= 6
        if has_argument:
            # A genuine argument can soften conviction slightly (Bayesian-ish)
            old = value.conviction
            value.conviction = round(max(0.3, value.conviction - 0.1), 3)
            value.updated_at = time.time()
            self.save()
            return {"changed": True, "old": old, "new": value.conviction,
                    "message": (f"That's a real argument — I've moved a little, "
                                f"but I still largely hold '{value_name}'.")}
        # Mere pressure: hold the line, and harden
        value.pressure_count += 1
        value.conviction = round(min(0.99, value.conviction + 0.02), 3)
        self.save()
        return {"changed": False, "pressure_count": value.pressure_count,
                "message": (f"I still disagree. Pushing harder won't move me — "
                            f"give me a real reason and I'll genuinely reconsider.")}

    def stats(self) -> dict:
        return {"values": {v.name: v.conviction for v in self.values.values()},
                "total_pressure_resisted": sum(v.pressure_count
                                               for v in self.values.values())}
