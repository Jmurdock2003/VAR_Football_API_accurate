import cv2
import numpy as np
from .detector import Detector
from .player_tracker import PlayerTracker
from .ball_tracker import BallTracker
from .team_assign import TeamAssigner
from .player_ball_assign import PlayerBallAssigner
from .event_detector.Event_Detecor import EventDetector
from .pitch_detector import PitchDetector
from .instance_detecor.Ball_Kick_Detector import BallKickDetector
from utils.bbox_utils import get_centre

class LiveProcessor:
    def __init__(self, source=0, detect_every: int = 1, attacking_dir='right'):
        self.cap = cv2.VideoCapture(source)
        self.detector = Detector()
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.team_assigner = TeamAssigner()
        self.ball_assigner = PlayerBallAssigner()
        self.event_detector = EventDetector(frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.pitch_detector = PitchDetector()
        self.kick_detector = BallKickDetector()

        self.frame_count = 0
        self.detect_every = detect_every
        self.last_dets = []
        self.last_ball = None
        self.last_player_possession = None
        self.halftime_mode = False
        self.team_1_dir = attacking_dir
        self.team_2_dir = 'left' if attacking_dir == 'right' else 'right'

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self.process(frame)

    def toggle_halftime(self):
        self.halftime_mode = not self.halftime_mode
        self.team_1_dir = 'left' if self.team_1_dir == 'right' else 'right'
        self.team_2_dir = 'left' if self.team_2_dir == 'right' else 'right'

    def process(self, frame):
        self.frame_count += 1
        fid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if self.halftime_mode:
            return {"frame_id": fid, "tracks": [], "pitch_lines": []}

        if self.frame_count % self.detect_every == 0:
            raw_dets = self.detector(frame)
            self.last_dets = [{
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cls": str(int(cls)),
                "id": None
            } for x1, y1, x2, y2, cls, conf in raw_dets]

        # Step 2: update trackers
        player_tracks = self.player_tracker.update(self.last_dets, frame)
        ball_tracks = self.ball_tracker.update(frame, self.last_dets)
        tracks = player_tracks + ball_tracks

        for track in tracks:
            track['cls'] = str(track.get('cls', '2'))

        # Step 3: Assign team colours
        team_tracks = self.team_assigner.assign(frame, tracks)

        # Step 4: Ball handling and event logic
        ball = next((t for t in team_tracks if t['cls'] == '0'), None)
        if ball:
            ball_center = get_centre(ball['bbox'])
            is_kicked = self.kick_detector.update(ball_center, self.frame_count)
            player_with_ball = self.ball_assigner.assign_ball_to_player(team_tracks, ball['bbox'])

            ball['possessed_by'] = player_with_ball
            ball['kicked'] = is_kicked

            if player_with_ball != -1:
                self.last_player_possession = player_with_ball

            print(f"Ball kicked: {is_kicked}")
            print(f"Ball possessed by player {player_with_ball}")

        #pitch_lines = self.pitch_detector.detect(frame)
        #print(pitch_lines)

        event, event_text = self.event_detector.detect(
            self.frame_count,
            team_tracks,
            ball,
            direction=self.team_1_dir,
            last_player_possession=self.last_player_possession
        )

        if event:
            print(f"[EVENT] {event_text}")

        return {
            "frame_id": fid,
            "tracks": team_tracks,
        }
