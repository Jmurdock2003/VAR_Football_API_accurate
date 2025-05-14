import cv2  # OpenCV for video writing
import os   # File system operations
from collections import deque  # Efficient fixed-length buffer

class ReplayBuffer:
    """
    Maintains a buffer of recent frames and saves clips around events.
    """
    def __init__(self, fps, clip_dir="clips", buffer_seconds=10):
        # Frames per second of input video stream
        self.fps = fps
        # Calculate maximum number of frames to buffer
        self.buffer_size = int(fps * buffer_seconds)
        # Rolling buffer of the last `buffer_size` frames
        self.frames = deque(maxlen=self.buffer_size)
        # Directory where event clips will be saved
        self.clip_dir = clip_dir
        os.makedirs(clip_dir, exist_ok=True)

    def add_frame(self, frame):
        """
        Add a new frame to the buffer. Copies the frame to avoid aliasing.
        """
        if frame is None:
            # Broken: frame may be None if capture failed; skip adding
            print("[ReplayBuffer] ⚠ Tried to add None frame to buffer.")
            return
        # Append a copy to preserve original
        self.frames.append(frame.copy())

    def save_event_clip(self, post_frames, event_type, index=1):
        """
        Save a video clip consisting of buffered frames plus post-event frames.

        Args:
          post_frames (list): additional frames after the trigger
          event_type (str): name prefix for the clip file
          index (int): numeric suffix to avoid filename collisions

        Returns:
          str or None: path to saved clip, or None on failure
        """
        # Combine buffered frames with any post-event frames
        all_frames = list(self.frames) + (post_frames or [])
        if not all_frames:
            print("[ReplayBuffer] ⚠ No frames available to save.")
            return None

        # Construct output filename and path
        filename = f"{event_type}_{index}.mp4"
        filepath = os.path.join(self.clip_dir, filename)

        # Infer frame dimensions from first frame
        height, width = all_frames[0].shape[:2]

        # Use MP4 codec; NOTE: 'mp4v' may not be supported on all platforms (broken)"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            filepath,
            fourcc,
            float(self.fps),  # Ensure fps is float (VideoWriter may require int)
            (width, height)
        )
        # Check if writer opened successfully
        if not writer.isOpened():
            print(f"[ReplayBuffer] ⚠ VideoWriter failed to open for {filepath}.")
            return None

        # Write each frame to the output clip
        for f in all_frames:
            if f is None:
                # Broken: skip None frames
                continue
            writer.write(f)
        writer.release()

        print(f"[ReplayBuffer] Saved event clip: {filepath}")
        return filepath

    # BROKEN: No method to clear buffer after saving clip; buffer continues to grow until full
    # Potential improvement: self.frames.clear() or similar after saving
