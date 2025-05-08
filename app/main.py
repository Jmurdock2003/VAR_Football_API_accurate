import os, uuid, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from core.stream import LiveProcessor

app = FastAPI()

# Uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Static (HTML/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

last_video = None

@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global last_video
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Not a video")
    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path  = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(await file.read())
    print(f">>> [UPLOAD] saved {path}")
    last_video = path
    return {"filename": fname}

@app.get("/stream")
def stream():
    if not last_video:
        raise HTTPException(400, "No video uploaded")
    def gen():
        for payload in LiveProcessor(source=last_video):
            evt = {"frame_id": payload["frame_id"], "tracks": payload["tracks"]}
            yield "data: " + json.dumps(evt) + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
