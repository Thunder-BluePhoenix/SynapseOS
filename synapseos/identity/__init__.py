"""
SynapseOS Identity — the companion's own self in the world.

When you give SynapseOS its own Gmail, YouTube, Git, and social accounts, this
package is where that identity lives:

  CredentialStore — secrets from env vars or data/identity/credentials.json
                    (gitignored), never hardcoded; one lookup surface for all
                    tools and the action executor
  Identity        — the companion's public profile: name, the accounts it
                    owns, its handles, and a short persona/values statement
  Outbox          — the oversight your vision asked for: outbound messages
                    (email to customers, social posts) are DRAFTED here and
                    sent only after approval. Autonomy with a brake.

Nothing here invents an account. It manages the ones you hand it, with audit
and approval, so a system that can email and post has clear limits.
"""

from synapseos.identity.credentials import CredentialStore
from synapseos.identity.profile import Account, Identity
from synapseos.identity.outbox import Outbox, OutboxItem

__all__ = ["CredentialStore", "Account", "Identity", "Outbox", "OutboxItem"]
