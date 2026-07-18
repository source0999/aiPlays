from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aiplays.dashboard.state import DashboardState


def create_app(telemetry_path: Path) -> FastAPI:
    app = FastAPI(title="aiPlays Dashboard", docs_url=None, redoc_url=None)
    state = DashboardState(telemetry_path)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/api/telemetry")
    def telemetry() -> dict[str, object]:
        return state.snapshot()

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(state.snapshot())
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            return

    return app
