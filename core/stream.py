import cv2  # OpenCV for video capture and processing
import numpy as np  # Numerical operations


# Custom modules for detection, tracking, and events
from .detectors.object_detector import Detector
from .trackers.player_tracker import PlayerTracker
from .trackers.ball_tracker import BallTracker
from .assigners.team_assign import TeamAssigner
from .assigners.player_ball_assign import PlayerBallAssigner
from .event_detector.Event_Detecor import EventDetector
from .assigners.Ball_Kick_Detector import BallKickDetector
#from .replay_buffer_broken import ReplayBuffer
from utils.bbox_utils import get_centre
from .event_detector.Rule_Knowledge_Graph import RuleKnowledgeGraph

class LiveProcessor:
    """
    Handles live video processing:
      • Captures frames
      • Detects objects (players, ball)
      • Tracks motion
      • Assigns teams and ball possession
      • Detects events (kicks, goals)
      • Buffers for replay
    """
    def __init__(self, source=0, detect_every=1, attacking_dir='right'):
        # Initialize video capture and validate source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Basic settings and state
        self.frame_count = 0
        self.detect_every = max(1, int(detect_every))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.rules_graph = RuleKnowledgeGraph()
        self.rules_graph.visualize("rules.png")  # saved in the working directory

        # Initialize processing modules
        self.detector = Detector()  # object detector
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.team_assigner = TeamAssigner()
        self.ball_assigner = PlayerBallAssigner()
        self.event_detector = EventDetector(frame_width=width)
        self.kick_detector = BallKickDetector()

        # Replay buffer for saving clips, broken
        #self.replay_buffer = ReplayBuffer(fps=self.fps, buffer_seconds=8)

        # Flags and memory
        self.halftime_mode = False
        self.last_detections = []
        self.last_player_possession = None

        # Set attacking directions
        self.team_1_dir = attacking_dir
        self.team_2_dir = 'left' if attacking_dir == 'right' else 'right'

    def __iter__(self):
        # Make the processor iterable over frames
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Stream ended or cannot read frame.")
                break
            try:
                yield self.process(frame)
            except Exception as e:
                print(f"Frame processing error #{self.frame_count}: {e}")
                continue

    def toggle_halftime(self):
        # Switch sides at half-time
        self.halftime_mode = not self.halftime_mode
        self.team_1_dir = 'left' if self.team_1_dir == 'right' else 'right'
        self.team_2_dir = 'left' if self.team_2_dir == 'right' else 'right'

    def process(self, frame):
        """
        Process a single frame and return:
          • frame_id
          • tracked objects
          • detected events
        """
        self.frame_count += 1
        frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        

        # Skip processing during half-time
        if self.halftime_mode:
            return {"frame_id": frame_id, "tracks": [], "event": None}

        # Perform detection at configured interval
        if self.frame_count % self.detect_every == 0:
            try:
                detections = self.detector(frame)
            except Exception as e:
                print(f"Detection error at frame {self.frame_count}: {e}")
                detections = []

            # Format detections for trackers
            self.last_detections = []
            for det in detections:
                try:
                    x1, y1, x2, y2, cls, conf = det
                    self.last_detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "cls": str(int(cls)),
                        "conf": float(conf),
                        "id": None
                    })
                except Exception:
                    # Skip malformed detection
                    continue

        # Update player and ball trackers
        try:
            player_tracks = self.player_tracker.update(self.last_detections, frame)
        except Exception as e:
            print(f"Player tracking error: {e}")
            player_tracks = []

        try:
            ball_tracks = self.ball_tracker.update(frame, self.last_detections)
        except Exception as e:
            print(f"Ball tracking error: {e}")
            ball_tracks = []

        # Combine tracks and cast class labels to strings
        all_tracks = player_tracks + ball_tracks
        for t in all_tracks:
            t['cls'] = str(t.get('cls', '2'))

        # Assign teams based on color or position
        try:
            team_tracks = self.team_assigner.assign(frame, all_tracks)
        except Exception as e:
            print(f"Team assignment error: {e}")
            team_tracks = all_tracks

        # Identify the ball and assign possession
        ball = next((t for t in team_tracks if t['cls'] == '0'), None)
        if ball:
            try:
                player_with_ball = self.ball_assigner.assign_ball_to_player(team_tracks, ball['bbox'])
            except Exception as e:
                print(f"Ball-to-player assigner error: {e}")
                player_with_ball = -1

            ball['possessed_by'] = int(player_with_ball)
            try:
                kicked = self.kick_detector.update(
                    ball,
                    next((p for p in team_tracks if p['id'] == player_with_ball), None),
                    self.frame_count
                )
                ball['kicked'] = bool(kicked)
            except Exception as e:
                print(f"Kick detection error: {e}")
                ball['kicked'] = False

        # Event detection (goals, fouls, etc.)
        try:
            event, event_text = self.event_detector.detect(
                self.frame_count,
                team_tracks,
                ball,
                direction=self.team_1_dir,
                last_player_possession=self.last_player_possession
            )
        except Exception as e:
            print(f"Event detection error: {e}")
            event, event_text = None, None

        # Prepare tracks for JSON serialization

        #adding an example of the TTS
        if self.frame_count == 15:
            event = "System Started"
            event_text = "System Started"

        for t in team_tracks:
            t['id'] = int(t.get('id', -1))
            t['bbox'] = [float(x) for x in t.get('bbox', [0,0,0,0])]
            t['velocity'] = [float(v) for v in t.get('velocity', [0.0,0.0])]
            t['team'] = int(t['team']) if t.get('team') is not None else None
            t['color'] = [int(c) for c in t.get('color', (128,128,128))]
            if 'kicked' in t:
                t['kicked'] = bool(t['kicked'])
            if 'possessed_by' in t:
                t['possessed_by'] = int(t['possessed_by'])

        # Return structured output
        return {
            "frame_id": frame_id,
            "tracks": team_tracks,
            "event": event,
            "event_text": event_text
        }
