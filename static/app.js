const fileInput       = document.getElementById("fileInput");
const uploadBtn       = document.getElementById("uploadBtn");
const pauseBtn        = document.getElementById("pauseBtn");
const halftimeBtn     = document.getElementById("halftimeBtn");
const directionSelect = document.getElementById("direction");
const status          = document.getElementById("status");
const errorDiv        = document.getElementById("error");
const video           = document.getElementById("video");
const canvas          = document.getElementById("canvas");
const ctx             = canvas.getContext("2d");

let es, paused = false, finished = false, matchPhase = "first";
const dets = {};
const FPS = 30;

/** Display an error message to the user and speak it via TTS */
function showError(message) {
  console.error(message);
  errorDiv.textContent = message;
  // Use browser TTS to announce
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.lang = 'en-GB';
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

/** Clear any previous error message */
function clearError() {
  errorDiv.textContent = "";
}

uploadBtn.addEventListener("click", async () => {
  clearError();
  if (!fileInput.files.length) {
    showError("Please select a video file to upload.");
    return;
  }
  status.textContent = "Uploading…";

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("direction", directionSelect.value);

  try {
    const resp = await fetch("/upload", { method: "POST", body: fd });
    if (!resp.ok) {
      throw new Error(`Upload failed (${resp.status})`);
    }
    const { filename } = await resp.json();
    status.textContent = `Uploaded ${filename}.`;
    video.src = `/uploads/${filename}`;

    video.onloadedmetadata = () => {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
      matchPhase = "first";
      halftimeBtn.textContent = "Halftime";
      halftimeBtn.disabled = false;
      finished = false;
      startStream();
    };
  } catch (err) {
    showError(err.message);
  }
});

pauseBtn.addEventListener("click", () => {
  clearError();
  paused = !paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  if (paused && es) {
    es.close();
  } else if (!paused && !finished) {
    startStream();
  }
});

halftimeBtn.addEventListener("click", async () => {
  clearError();
  try {
    await fetch("/halftime", { method: "POST" });
  } catch (err) {
    showError("Failed to toggle halftime: " + err.message);
    return;
  }

  if (matchPhase === "first") {
    if (es) es.close();
    matchPhase = "halftime";
    halftimeBtn.textContent = "Start 2nd Half";
    status.textContent = "Halftime — detections paused.";
  } else if (matchPhase === "halftime") {
    startStream();
    matchPhase = "second";
    halftimeBtn.textContent = "Full Time";
    status.textContent = "Second half started — detections resumed.";
  } else if (matchPhase === "second") {
    if (es) es.close();
    paused = true;
    finished = true;
    halftimeBtn.disabled = true;
    status.textContent = "Thank you for using Murdock VAR system";
  }
});

function startStream() {
  clearError();
  if (es) es.close();

  status.textContent = "Streaming frames…";
  es = new EventSource("/stream");

  es.onmessage = e => {
    if (paused || finished) return;
    let p;
    try {
      p = JSON.parse(e.data);
    } catch (err) {
      showError("Malformed stream data: " + err.message);
      return;
    }
    dets[p.frame_id] = p.tracks;
    const t = (p.frame_id - 1) / FPS;
    video.currentTime = t;

    // Speak event text if available
    if (p.event_text) {
      const utterance = new SpeechSynthesisUtterance(p.event_text);
      utterance.lang = 'en-GB';
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
    }
  };

  es.onerror = err => {
    showError("Stream connection error.");
    if (es) es.close();
  };
}

video.onseeked = () => {
  clearError();
  if (paused || finished) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  const frameId = Math.round(video.currentTime * FPS) + 1;
  const tracks  = dets[frameId] || [];

  if (matchPhase === "halftime") return;

  let possessedPlayerId = -1;
  const ball = tracks.find(t => t.cls === "0" && "possessed_by" in t);
  if (ball) possessedPlayerId = ball.possessed_by;

  tracks.forEach(t => {
    const [x1, y1, x2, y2] = t.bbox;
    const [r, g, b] = t.color || [128, 128, 128];
    const col = `rgb(${r}, ${g}, ${b})`;

    ctx.strokeStyle = col;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = col;
    ctx.font = "14px Arial";
    ctx.fillText(`ID:${t.id} T${t.team}`, x1, y2 + 15);

    if (t.id === possessedPlayerId && t.cls === "2") {
      const cx = (x1 + x2) / 2;
      const cy = y1 - 10;
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  });
};

// Global JS error handler
window.addEventListener("error", event => {
  showError("An unexpected error occurred: " + event.message);
});
