from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from starlette.websockets import WebSocketDisconnect

import uvicorn
import os
import json
from src.vision_detector import VisionDetector
from src.page_locator import FixedRectangleLocator, RectangleLocator
from src.image_classifier import ClassicVisionClassifier

import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor


PORT = int(os.environ.get("WS_PORT", 6969))

refs = [
    "darkness",
    "discover",
    "enlightenment"
]

IMG_REFS = [f"static/chapter_ref/{ref}.jpg" for ref in refs]
VIDEOS = [f"static/videos/{ref}.mp4" for ref in refs]

classifier = ClassicVisionClassifier(reference_images=IMG_REFS, conf_ths=50)
# detector = RectangleLocator(classifier=classifier, refresh_rate=1, conf_frames=10)
default_rect = (0.1, 0.1, 0.85, 0.8)  # Example rectangle coordinates
locator = FixedRectangleLocator(default_rect)
detector = VisionDetector(locator, classifier, refresh_rate=1, conf_frames=10)

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
    return json.dumps({
        "index": index,
        "chapter": refs[index], 
        "path": "/" + VIDEOS[index].replace("\\", "/")
    })

from starlette.websockets import WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_index = 0
    await websocket.send_text(socket_message(current_index))

    winner_queue = queue.Queue()
    stop_event = threading.Event()

    def detector_worker():
        for winner in detector.run():
            if stop_event.is_set():
                break
            # Map winner (string) to index if needed
            if winner is not None:
                try:
                    # depends on winner output type.
                    winner_index = refs.index(winner) if isinstance(winner, str) else winner
                except ValueError:
                    winner_index = None
            else:
                winner_index = None
            winner_queue.put(winner_index)
        stop_event.set()  # Signal end of processing

    worker_thread = threading.Thread(target=detector_worker, daemon=True)
    worker_thread.start()

    try:
        loop = asyncio.get_event_loop()
        while True:
            winner = await loop.run_in_executor(None, winner_queue.get)
            if winner is None:
                break
            if not isinstance(winner, int) or winner < 0 or winner >= len(VIDEOS):
                await websocket.send_text(json.dumps({"error": "Invalid video index"}))
                continue
            if winner != current_index:
                current_index = winner
                await websocket.send_text(socket_message(current_index))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        stop_event.set()
        worker_thread.join(timeout=1)

if __name__ == "__main__":
    uvicorn.run("main:a pp", host="0.0.0.0", port=PORT, reload=True, log_level="debug")
