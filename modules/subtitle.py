import asyncio
import json
import threading
import time
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol


class JsonWebSocketPublisher:
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.queue: Optional[asyncio.Queue[str]] = None
        self.clients: Set[WebSocketServerProtocol] = set()

    async def _handler(self, websocket: WebSocketServerProtocol):
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)

    async def _safe_send(self, ws: WebSocketServerProtocol, msg: str):
        try:
            await ws.send(msg)
        except Exception:
            self.clients.discard(ws)

    async def _broadcast_loop(self):
        if self.queue is None:
            self.queue = asyncio.Queue()
        while True:
            msg = await self.queue.get()
            if msg is None:
                break
            if self.clients:
                await asyncio.gather(*(self._safe_send(ws, msg) for ws in list(self.clients)))

    async def _run(self):
        self.queue = asyncio.Queue()
        async with websockets.serve(self._handler, self.host, self.port):
            await self._broadcast_loop()

    def start(self):
        if self.thread is not None:
            return

        def runner():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run())

        self.thread = threading.Thread(target=runner, daemon=True)
        self.thread.start()
        while self.queue is None:
            time.sleep(0.01)

    def stop(self):
        if self.loop and self.queue:
            def stopper():
                self.queue.put_nowait(None)
            self.loop.call_soon_threadsafe(stopper)
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            self.loop = None
            self.queue = None
            self.clients.clear()

    def send(self, payload: Dict[str, Any]):
        if not self.loop or not self.queue:
            raise RuntimeError("WebSocket publisher not started")
        msg = json.dumps(payload, ensure_ascii=False)

        def sender():
            self.queue.put_nowait(msg)
        self.loop.call_soon_threadsafe(sender)


_publisher: Optional[JsonWebSocketPublisher] = None


def start_subtitle_server(host: str = "localhost", port: int = 8765) -> JsonWebSocketPublisher:
    global _publisher
    if _publisher is None:
        _publisher = JsonWebSocketPublisher(host, port)
        _publisher.start()
    return _publisher


def send_subtitle(data: Dict[str, Any]):
    if _publisher is None:
        raise RuntimeError("Subtitle server not started")
    _publisher.send(data)


def stop_subtitle_server():
    global _publisher
    if _publisher is not None:
        _publisher.stop()
        _publisher = None


if __name__ == "__main__":
    start_subtitle_server()
    i = 0
    try:
        while True:
            payload = {
                "time": i,
                "text": f"test subtitle {i}",
                "labels": {"speaker": "S0"},
                "prosody": {"loudness": 0.5, "arousal": 0.5, "valence": 0.5},
            }
            send_subtitle(payload)
            i += 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_subtitle_server()
