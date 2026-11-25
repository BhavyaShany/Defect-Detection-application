"""WebSocket connection manager and event helpers.

Event schema (all JSON objects have ``type`` and ``ts``):

    hello     {server_version}                          sent on connect
    pong      {}                                        reply to a client "ping"
    progress  {scan_id, stage, message, filename}       stage: received | parsing | preprocessing |
                                                        inference | saving
    result    {scan_id, scan: <summary>}                inspection finished and stored
    failed    {scan_id, code, message, filename}        inspection rejected / failed
"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Set

from fastapi import WebSocket

log = logging.getLogger("defect.ws")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()

    @property
    def count(self) -> int:
        return len(self._clients)

    async def connect(self, ws: WebSocket, server_version: str) -> None:
        await ws.accept()
        self._clients.add(ws)
        await self._send(ws, {"type": "hello", "server_version": server_version})

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def _send(self, ws: WebSocket, message: Dict[str, Any]) -> bool:
        try:
            await asyncio.wait_for(ws.send_text(json.dumps({**message, "ts": _now()})), timeout=5)
            return True
        except Exception:
            self._clients.discard(ws)
            return False

    async def broadcast(self, message: Dict[str, Any]) -> None:
        if not self._clients:
            return
        await asyncio.gather(*(self._send(ws, message) for ws in list(self._clients)))

    async def pong(self, ws: WebSocket) -> None:
        await self._send(ws, {"type": "pong"})


manager = ConnectionManager()
