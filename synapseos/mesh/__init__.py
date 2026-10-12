"""
SynapseOS Device Mesh — offline-first peer memory sync.

The same companion, on every device, with the same learned facts, same open
questions, same emotional state. No cloud required; devices sync directly when
they can reach each other. Offline-safe: writes queue and flush on reconnect.

Core design:
  SyncStore   — LWW oplog; deterministic merge; any order, any subset of peers
  MeshNode    — one device; owns a SyncStore + list of peer URLs
  Transport   — InProcessTransport (tests) or HttpTransport (real devices)

What syncs: semantic facts (deletions as tombstones; incoming facts pass
through the belief reviser), wonders (including updates), episode summaries,
emotional state, aesthetic taste, values and reflection snapshots.

Security: HttpTransport signs every request and response with HMAC-SHA256
(shared key), enforces a peer-node allowlist, rejects stale timestamps and
replayed nonces, and refuses non-loopback traffic without TLS. There is no
unauthenticated mode. The controller auto-starts sync when enable_mesh=True
and mesh_shared_key + mesh_allowed_peer_ids (or an explicit transport) are set.

Wiring:
  from synapseos.mesh import SyncStore, MeshNode, InProcessTransport, HttpTransport
"""

from synapseos.mesh.sync_store import SyncStore, SyncEntry
from synapseos.mesh.transport import InProcessTransport, HttpTransport
from synapseos.mesh.mesh_node import MeshNode

__all__ = ["SyncStore", "SyncEntry", "MeshNode", "InProcessTransport", "HttpTransport"]
