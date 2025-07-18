from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

import uvicorn
import os
import json
from src.fake_detector import video_detector

PORT = int(os.environ.get("WS_PORT", 6969))

VIDEOS = [
    "static/videos/video1.mp4",
    "static/videos/video2.mp4",
    "static/videos/video3.mp4"
]

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Serve static files (JS, videos, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/web_client.html")
async def web_client(request: Request):
    ws_port = int(os.environ.get("WS_PORT", 6969))
    default_video_path = VIDEOS[0]
    return templates.TemplateResponse("web_client_template.html", {"request": request, "ws_port": ws_port, "default_video_path": default_video_path})

def socket_message(index: int) -> str:
    """
    Helper function to format the message sent over the WebSocket.
    """
    return json.dumps({"index": index, "path": "/" + VIDEOS[index].replace("\\", "/")})

from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_index = 0
    try:
        # Send the default (first) video index
        await websocket.send_text(socket_message(current_index))
        while True:
            try:
                async for idx in video_detector():
                    if not isinstance(idx, int) or idx < 0 or idx >= len(VIDEOS):
                        await websocket.send_text(json.dumps({"error": "Invalid video index"}))
                        continue
                    if idx != current_index:
                        current_index = idx
                        await websocket.send_text(socket_message(current_index))
            except WebSocketDisconnect:
                break
            except Exception as e:
                try:
                    await websocket.send_text(json.dumps({"error": str(e)}))
                except Exception:
                    pass
                break
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True, log_level="debug")
