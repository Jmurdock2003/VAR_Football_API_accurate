const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const pauseBtn  = document.getElementById("pauseBtn");
const status    = document.getElementById("status");
const video     = document.getElementById("video");
const canvas    = document.getElementById("canvas");
const ctx       = canvas.getContext("2d");

let es, paused = false;
const dets = {};
const FPS = 30;

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return alert("Select a video.");
  status.textContent = "Uploading…";
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  const resp = await fetch("/upload", { method: "POST", body: fd });
  if (!resp.ok) {
    status.textContent = `Upload failed (${resp.status})`;
    return;
  }
  const { filename } = await resp.json();
  status.textContent = `Uploaded ${filename}.`;
  video.src = `/uploads/${filename}`;
  video.onloadedmetadata = () => {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    startStream();
  };
});

pauseBtn.addEventListener("click", () => {
  paused = !paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  if (paused && es) {
    es.close();
  } else if (!paused) {
    startStream();
  }
});

function startStream() {
  if (es) es.close();
  status.textContent = "Streaming frames…";
  es = new EventSource("/stream");

  es.onmessage = e => {
    if (paused) return;
    const p = JSON.parse(e.data);
    dets[p.frame_id] = p.tracks;
    const t = (p.frame_id - 1) / FPS;
    video.currentTime = t;
  };

  video.onseeked = () => {
    if (paused) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    const frameId = Math.round(video.currentTime * FPS) + 1;
    const tracks  = dets[frameId] || [];
    tracks.forEach(t => {
      const [x1, y1, x2, y2] = t.bbox;
      const [r, g, b] = t.color || [128, 128, 128];
      const col = `rgb(${r}, ${g}, ${b})`;
      ctx.strokeStyle = col;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = col;
      ctx.font = "14px Arial";
      ctx.fillText(`ID:${t.id} T${t.team}`, x1, y2 + 15);
    });
  };

  es.onerror = () => {
    status.textContent = "Stream error";
    if (es) es.close();
  };
}
