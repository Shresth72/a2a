import json
import uuid
import httpx
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from typing import Any, AsyncIterable, List

from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)

from remote_agent_connection import RemoteAgentConnection

load_dotenv()


class HostAgent:

    def __init__(self):
        self.remote_agent_connection: dict[str, RemoteAgentConnection] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self_user_id = "host_agent"
        self._runner = Runner(
            app_name=self._agent,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnection(
                        agent_card=card, agent_url=address
                    )

                    self.remote_agent_connection[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")
