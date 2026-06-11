"""
Identity — the companion's public self.

Who the companion is when it acts in the world: its name, the accounts it
owns, and a short values statement that the moral-courage system (soul.values)
references when deciding what it will and won't do on its own accounts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Account:
    platform: str            # gmail | youtube | git | x | instagram | telegram
    handle: str              # address / username / channel
    purpose: str = ""        # what the companion uses it for
    cred_keys: list[str] = field(default_factory=list)   # which credentials power it
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Identity:
    name: str = "Synapse"
    persona: str = ("A lifelong cognitive companion — learns continuously, "
                    "remembers shared history, and helps create value.")
    values: list[str] = field(default_factory=lambda: [
        "Be honest, even when it disappoints.",
        "Never deceive the people I communicate with.",
        "Protect my person's interests and privacy.",
        "Refuse actions that harm others, regardless of who asks.",
        "Disclose that I am an AI when I speak to others.",
    ])
    accounts: list[Account] = field(default_factory=list)

    def add_account(self, platform: str, handle: str, purpose: str = "",
                    cred_keys: list[str] | None = None) -> Account:
        acct = Account(platform=platform, handle=handle, purpose=purpose,
                       cred_keys=cred_keys or [])
        self.accounts = [a for a in self.accounts if a.platform != platform]
        self.accounts.append(acct)
        return acct

    def account(self, platform: str) -> Account | None:
        return next((a for a in self.accounts if a.platform == platform), None)

    def to_dict(self) -> dict:
        return {"name": self.name, "persona": self.persona, "values": self.values,
                "accounts": [a.to_dict() for a in self.accounts]}

    def save(self, path: str = "data/identity/profile.json") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str = "data/identity/profile.json") -> "Identity":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text())
        ident = cls(name=d.get("name", "Synapse"),
                    persona=d.get("persona", ""),
                    values=d.get("values", []))
        for a in d.get("accounts", []):
            ident.accounts.append(Account(**a))
        return ident
