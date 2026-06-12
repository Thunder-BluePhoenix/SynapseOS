"""
SynapseOS Device Mesh — offline-first peer memory sync.

The same companion, on every device, with the same learned facts, same open
questions, same emotional state. No cloud required; devices sync directly when
they can reach each other. Offline-safe: writes queue and flush on reconnect.

Core design:
  SyncStore   — LWW oplog; deterministic merge; any order, any subset of peers
  MeshNode    — one device; owns a SyncStore + list of peer URLs
  Transport   — InProcessTransport (tests) or HttpTransport (real devices)

Wiring:
  from synapseos.mesh import SyncStore, MeshNode, InProcessTransport, HttpTransport
"""

from synapseos.mesh.sync_store import SyncStore, SyncEntry
from synapseos.mesh.transport import InProcessTransport, HttpTransport
from synapseos.mesh.mesh_node import MeshNode

__all__ = ["SyncStore", "SyncEntry", "MeshNode", "InProcessTransport", "HttpTransport"]
