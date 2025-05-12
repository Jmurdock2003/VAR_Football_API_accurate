import cv2
import numpy as np
from .detector import Detector
from .tracker import Tracker
from .team_assign import TeamAssigner
from .player_ball_assign import PlayerBallAssigner
from .event_detector.Event_Detecor import EventDetector
from .pitch_detector import PitchDetector
from .instance_detecor.Ball_Kick_Detector import BallKickDetector
from utils.bbox_utils import get_centre





class LiveProcessor:
    def __init__(self, source=0, detect_every: int = 1, attacking_dir='right'):
        self.cap            = cv2.VideoCapture(source)
        self.detector       = Detector()
        self.tracker        = Tracker(max_age=15, n_init=2, max_cosine_distance=0.25)
        self.team_assigner  = TeamAssigner()
        self.ball_assigner  = PlayerBallAssigner()
        self.last_player_possession = None
        self.detect_every   = detect_every
        self.frame_count    = 0
        self.last_dets      = []
        self.last_ball      = None
        self.team_1_dir     = attacking_dir
        self.team_2_dir     = 'left' if attacking_dir == 'right' else 'right'
        self.event_detector = EventDetector(frame_width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.halftime_mode  = False
        self.pitch_detector = PitchDetector()
        self.kick_detector = BallKickDetector()


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
            # Show frame ID without detections
            return {"frame_id": fid, "tracks": []}

        if self.frame_count % self.detect_every == 0:
            raw_dets = self.detector(frame)
            formatted_dets = []
            for det in raw_dets:
                x1, y1, x2, y2, cls, conf = det
                formatted_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": str(int(cls)),
                    "id": None
                })
            self.last_dets = formatted_dets

        dets = self.last_dets
        tracks = self.tracker.update(dets, frame)

        for track in tracks:
            track['cls'] = str(track.get('cls', '2'))

        team_tracks = self.team_assigner.assign(frame, tracks)
        team_tracks, self.last_ball = self._suppress_duplicate_balls(team_tracks, self.last_ball)

        ball = next((t for t in team_tracks if t['cls'] == '0'), None)

        if ball:
            ball_center = get_centre(ball['bbox'])
            player_with_ball = self.ball_assigner.assign_ball_to_player(team_tracks, ball['bbox'])
            is_kicked = self.kick_detector.update(ball_center, self.frame_count)
            if player_with_ball != -1:
                self.last_player_possession = player_with_ball
            ball['possessed_by'] = player_with_ball
            ball['kicked'] = is_kicked
            print(f"Ball kicked:{is_kicked}")
            print(f"Ball possessed by player {player_with_ball}")
        
        pitch_lines = self.pitch_detector.detect(frame)
        print (pitch_lines)

        event, event_text = self.event_detector.detect(
            self.frame_count,
            team_tracks,
            ball,
            direction=self.team_1_dir,  # or however you pass direction
            last_player_possession=self.last_player_possession
        )

        if event:
            print(f"[EVENT] {event_text}")


        return {
            "frame_id": fid,
            "tracks": team_tracks,
            "pitch_lines": pitch_lines
        }


    def _suppress_duplicate_balls(self, tracks, prev_ball):
        balls = [t for t in tracks if t['cls'] == '0']
        if not balls:
            return tracks, None

        balls.sort(key=lambda b: float(b.get('conf', 0)), reverse=True)

        if prev_ball:
            px, py = get_centre(prev_ball['bbox'])
            balls.sort(key=lambda b: np.linalg.norm(
                np.array(get_centre(b['bbox'])) - np.array([px, py])
            ))

        best_ball = balls[0]
        filtered = [t for t in tracks if t['cls'] != '0'] + [best_ball]
        return filtered, best_ball
