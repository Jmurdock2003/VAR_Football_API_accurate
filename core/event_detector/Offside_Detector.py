from event_detector.Rule_Knowledge_Graph import RuleKnowledgeGraph


class OffsideDetector:
    def __init__(self):
        self.kg = RuleKnowledgeGraph()

    def check_offside(self, player_positions, ball_position, defenders, frame_width, attack_direction):
        if len(defenders) < 2 or not player_positions:
            return None

        # Sort defenders by x-position to find second last defender
        defenders_sorted = sorted(defenders, key=lambda p: p[1][0], reverse=(attack_direction == "left"))
        second_last_def_x = defenders_sorted[1][1][0]

        # Find attacker closest to ball
        min_dist = float("inf")
        receiver = None
        for player_id, (x, y) in player_positions:
            dist = ((x - ball_position[0])**2 + (y - ball_position[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                receiver = (player_id, x)

        if receiver is None or min_dist > 50:
            return None  # No eligible attacker close enough

        player_id, player_x = receiver
        centre_x = frame_width // 2

        # Apply rule conditions from knowledge graph
        conditions = self.kg.get_conditions("Offside Rule")

        player_in_opponent_half = (
            (attack_direction == "right" and player_x > centre_x) or
            (attack_direction == "left" and player_x < centre_x)
        )
        ahead_of_ball = (
            (attack_direction == "right" and player_x > ball_position[0]) or
            (attack_direction == "left" and player_x < ball_position[0])
        )
        ahead_of_defender = (
            (attack_direction == "right" and player_x > second_last_def_x) or
            (attack_direction == "left" and player_x < second_last_def_x)
        )

        if all([player_in_opponent_half, ahead_of_ball, ahead_of_defender]):
            return player_id

        return None
