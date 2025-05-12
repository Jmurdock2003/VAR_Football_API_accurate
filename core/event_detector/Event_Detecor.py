import numpy as np
from .Offside_Detector import OffsideDetector
from .ThrowIn_Detector import ThrowInDetector
from .CornerGoal_Detector import CornerGoalDetector
from utils.bbox_utils import get_centre

class EventDetector:
    def __init__(self, frame_width):
        self.offside = OffsideDetector()
        #self.throwin = ThrowInDetector()
        #self.corner  = CornerGoalDetector()
        self.frame_width = frame_width
        self.last_event = None
        self.waiting_for_possession = False
        self.pending_offside_list = []

    def detect(self, frame_id, tracks, ball, direction, last_player_possession=None):
        possessing_player = next((p for p in tracks if p['id'] == last_player_possession), None)
        possessing_team = possessing_player['team'] if possessing_player else None

        attackers = [t for t in tracks if t['cls'] == '2' and t['team'] == possessing_team]
        defenders = [t for t in tracks if t['cls'] == '2' and t['team'] not in [None, 0, possessing_team]]

        ball_position = get_centre(ball['bbox']) if ball else None
        event = None
        event_text = None

        if ball_position:
            # Offside update candidates each frame
            self.offside.update_candidates(attackers, defenders, ball_position, direction, self.frame_width)

            if ball['kicked'] == True:
                self.pending_offside_list = self.offside.offside_candidates.copy()
                self.waiting_for_possession = True

            if self.waiting_for_possession and 'possessed_by' in ball and ball['possessed_by'] != -1:
                for pid in self.pending_offside_list:
                    if ball['possessed_by'] == pid:
                        event = 'Offside'
                        event_text = f"Offside by Player {pid}"
                        self.waiting_for_possession = False
                        break
                else:
                    self.waiting_for_possession = False  # possession occurred but not offside

            # Throw-In Detection
            '''elif self.throwin.check_throw_in(ball_position, self.frame_width):
                event = 'Throw-In'
                event_text = f"Ball out for a Throw-In"'''

            # Corner or Goal Kick Detection
            '''

            else:
            
                result = self.corner.check_corner_goal(ball_position, defenders, self.frame_width)
                if result:
                    event = result  # 'Corner' or 'Goal Kick'
                    event_text = f"Ball out for a {result}"
            '''

        self.last_event = (event, event_text)
        return event, event_text

