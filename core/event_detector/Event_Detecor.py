import numpy as np
from .Offside_Detector import OffsideDetector
# from .ThrowIn_Detector import ThrowInDetector
# from .CornerGoal_Detector import CornerGoalDetector
from utils.bbox_utils import get_centre

class EventDetector:
    def __init__(self, frame_width):
        self.offside = OffsideDetector()
        # self.throwin = ThrowInDetector()
        # self.corner  = CornerGoalDetector()
        self.frame_width = frame_width
        self.last_event = None
        self.waiting_for_possession = False
        self.pending_offside_list = []
        self.last_kick_frame = -1

    def detect(self, frame_id, tracks, ball, direction, last_player_possession):
        possessing_player = None
        if ball and isinstance(ball, dict) and 'possessed_by' in ball and isinstance(ball['possessed_by'], int):
            possessing_player = next((p for p in tracks if p['id'] == ball['possessed_by']), None)

        possessing_team = possessing_player['team'] if possessing_player else None

        attackers = [t for t in tracks if t['cls'] == '2' and t['team'] == possessing_team]
        defenders = [t for t in tracks if t['cls'] == '2' and t['team'] not in [None, 0, possessing_team]]

        ball_position = get_centre(ball['bbox']) if ball else None
        event = None
        event_text = None

        if ball_position:
            self.offside.update_candidates(attackers, defenders, ball_position, direction, self.frame_width)
            print(f"Offside candidates: {self.offside.offside_candidates}")
            # External kick detection sets this flag
            if ball.get('kicked', False):
                self.pending_offside_list = self.offside.offside_candidates.copy()
                self.waiting_for_possession = True
                self.last_ball_holder = last_player_possession

            # If a new player gains possession, check for offside
            if self.waiting_for_possession and 'possessed_by' in ball and ball['possessed_by'] != -1:
                if ball['possessed_by'] != self.last_ball_holder:
                    for pid, _ in self.pending_offside_list:
                        if ball['possessed_by'] == pid:
                            event = 'Offside'
                            event_text = f"Offside by Player {pid}"
                            break
                self.waiting_for_possession = False

            # Throw-In Detection (placeholder)
            # elif self.throwin.check_throw_in(ball_position, self.frame_width):
            #     event = 'Throw-In'
            #     event_text = f"Ball out for a Throw-In"

            # Corner or Goal Kick Detection (placeholder)
            # else:
            #     result = self.corner.check_corner_goal(ball_position, defenders, self.frame_width)
            #     if result:
            #         event = result
            #         event_text = f"Ball out for a {result}"

        self.last_event = (event, event_text)
        return event, event_text