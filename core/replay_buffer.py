import cv2
import os
from collections import deque

class ReplayBuffer:
    def __init__(self, fps, clip_dir="clips", buffer_seconds=10):
        """
        Keeps the last N seconds of frames and saves clips around events.
        :param fps: frames per second of the input video
        :param clip_dir: directory to save output clips
        :param buffer_seconds: number of seconds to retain in buffer
        """
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.frames = deque(maxlen=self.buffer_size)
        self.clip_dir = clip_dir
        os.makedirs(clip_dir, exist_ok=True)

    def add_frame(self, frame):
        """Adds a frame to the rolling buffer."""
        self.frames.append(frame.copy())

    def save_event_clip(self, post_frames, event_type, index=1):
        """
        Saves a video clip for a detected event.

        :param post_frames: list of additional frames to include after the event
        :param event_type: event name (e.g. 'offside', 'throwin')
        :param index: numeric identifier for the clip name
        :return: full path to saved file
        """
        all_frames = list(self.frames) + post_frames
        if not all_frames:
            print("[ReplayBuffer] ⚠ No frames available to save.")
            return None

        filename = f"{event_type}_{index}.mp4"
        filepath = os.path.join(self.clip_dir, filename)
        height, width = all_frames[0].shape[:2]
        writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))

        for f in all_frames:
            writer.write(f)
        writer.release()
        print(f"[ReplayBuffer] ✅ Saved event clip: {filepath}")
        return filepath