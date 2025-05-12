from .Rule_Knowledge_Graph import RuleKnowledgeGraph

class OffsideDetector:
    def __init__(self):
        self.kg = RuleKnowledgeGraph()
        self.offside_candidates = []
        self.last_kick_frame = -1
        self.last_kicked_by_team = None
        self.triggered_offside = None

    def update_candidates(self, attackers, defenders, ball_position, attack_dir, frame_width):
        self.offside_candidates = []

        if len(defenders) < 2:
            return

        sorted_defs = sorted(defenders, key=lambda d: self._get_far_side(d['bbox'], attack_dir), reverse=(attack_dir == 'left'))
        second_last_def_x = self._get_far_side(sorted_defs[1]['bbox'], attack_dir)
        ball_x = ball_position[0]

        conditions = self.kg.get_conditions("Offside")

        for player in attackers:
            player_x = self._get_far_side(player['bbox'], attack_dir)

            condition_results = {
                "Ball is played or touched by teammate": True,  # handled externally
                "Player is in opponent's half": (
                    (attack_dir == "right" and player_x > frame_width // 2) or
                    (attack_dir == "left" and player_x < frame_width // 2)
                ),
                "Player is ahead of the second-last defender": (
                    (attack_dir == "right" and player_x > second_last_def_x) or
                    (attack_dir == "left" and player_x < second_last_def_x)
                ),
                "Player is ahead of the ball": (
                    (attack_dir == "right" and player_x > ball_x) or
                    (attack_dir == "left" and player_x < ball_x)
                ),
                "Player interferes with play": True  # evaluated on possession
            }

            if all(condition_results.get(c, False) for c in conditions):
                self.offside_candidates.append((player['id'], player.get('team')))

    def check_violation(self, new_possessor_id):
        for pid, team in self.offside_candidates:
            if new_possessor_id == pid:
                self.triggered_offside = pid
                return pid
        return None

    def _get_far_side(self, bbox, direction):
        x1, _, x2, _ = bbox
        return max(x1, x2) if direction == "right" else min(x1, x2)
