# may-project-final

## Overview
This project detects book chapters using computer vision and plays matching videos for each chapter. It combines a FastAPI backend, a web client, and image processing techniques to deliver an interactive multimedia experience.

## Features
- Detects book chapters from images using a vision model.
- Plays corresponding videos for each detected chapter.
- Real-time communication between client and server via WebSocket.

## Components
1. **FastAPI Server**: Serves the web client, handles WebSocket connections, and manages backend logic.
2. **Web Client**: Displays videos, receives chapter detection results, and handles user interactions.
3. **Image Processing**: Uses various algorithms to detect book chapters from images.


### Running the Application
1. Start the FastAPI server:
   ```powershell
   python main.py
   ```
2. Optionally, you can create an env file with the following content:
   ```plaintext
   VIDEO_PORT=8000
   ```
   This sets the video port for the web client.
3. Open your browser and navigate to the provided URL (typically http://localhost:8000).

## Project Structure
- `main.py` - Entry point for the FastAPI server.
- `src/ - Image procsseing and chapter detection logic.
- `static/` - Static assets (images, videos).
- `templates/` - HTML templates for the web client.

## TODO
### Image Processing
Implement the following image comparison methods:
- [ ] imagehash
- [ ] ORB
- [ ] SSIM
