import random
import asyncio
    # Detector is only responsible for outputting the index. Video names/locations are managed by the server.
async def video_detector():
    """
    Async generator that yields a random video index every 5 seconds.
    """
    current_index = None
    while True:
        idx = random.randint(0, 2)
        yield idx
        await asyncio.sleep(5)
