import os
import uuid
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.stream import LiveProcessor  # Main processing pipeline for live video frames

# Initialize FastAPI application
app = FastAPI()

# Directory to store uploaded video files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve uploaded videos under /uploads URL path
app.mount(
    "/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads"
)
# Serve static assets (HTML, JS, CSS) under /static URL path
app.mount(
    "/static", StaticFiles(directory="static"), name="static"
)

# Global state to track the last uploaded video and its processor
last_video = None
processor = None

@app.get("/")
async def index():
    """
    Serve the main HTML page for the Murdock VAR System.
    """
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),  # Video file upload field
    direction: str = Form("right")  # Team 1 attacking direction
):
    """
    Handle video uploads:
      • Validate file is a video
      • Save with a unique filename
      • Initialize the LiveProcessor pipeline
    """
    global last_video, processor

    # Reject non-video content types
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a video.")

    # Generate unique filename to avoid collisions
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    # Write uploaded file to disk
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
    print(f"*** Uploaded and saved: {save_path}")

    # Store path and initialize processing pipeline
    last_video = save_path
    processor = LiveProcessor(source=last_video, attacking_dir=direction)

    # Return filename for client to construct video URL
    return {"filename": unique_name}

@app.get("/stream")
def stream():
    """
    Stream processed frame data via Server-Sent Events (SSE):
      • Each event contains JSON with frame_id, tracks, and events
    """
    if not processor:
        raise HTTPException(status_code=400, detail="No video uploaded yet.")

    def event_generator():
        # Iterate over processor yields until video ends
        for payload in processor:
            # Build a minimal event dict for client
            evt = {
                "frame_id": payload.get("frame_id"),
                "tracks": payload.get("tracks"),
                "event": payload.get("event"),
                "event_text": payload.get("event_text")
            }
            # SSE: data: <json>\n\n
            yield f"data: {json.dumps(evt)}\n\n"

    # Return streaming response with text/event-stream MIME type
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.post("/halftime")
def halftime():
    """
    Toggle halftime mode, flipping attacking directions.
    Used by client to pause/resume detection at half-time.
    """
    if processor:
        processor.toggle_halftime()
        return {"message": "Halftime toggled, directions switched."}

    # No active processing session
    raise HTTPException(status_code=400, detail="No active video session.")
