"""
SynapseOS Gateway — the companion's presence on messaging platforms.

The action layer (synapseos.actions.comms) lets the companion SEND messages.
This package lets it RECEIVE and reply — its own bot on Telegram and Discord,
reachable like a person:

  TelegramBot  — stdlib long-poll (getUpdates) + sendMessage, no dependency.
                 Every inbound message is routed through CognitiveController
                 .think(), so moral courage, affect, agents, memory, and the
                 whole mind apply to a chat the same as a local query.
  DiscordBot   — outbound via webhook (stdlib); inbound via discord.py when
                 installed, otherwise disabled with a clear note.
  MessageGateway — runs the bots as daemon threads, applies a chat allowlist
                 (only talk to people you approve), and audits every exchange.

Identity + oversight: the bot announces it is an AI on first contact, obeys
the chat allowlist, and logs every message in/out.
"""

from synapseos.gateway.telegram_bot import TelegramBot
from synapseos.gateway.discord_bot import DiscordBot
from synapseos.gateway.gateway import MessageGateway, GatewayConfig

__all__ = ["TelegramBot", "DiscordBot", "MessageGateway", "GatewayConfig"]
