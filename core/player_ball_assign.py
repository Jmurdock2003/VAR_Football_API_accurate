import sys
sys.path.append('../')  # adjust this path if needed

from utils.bbox_utils import get_centre, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70


    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_centre(ball_bbox)

        minimum_distance = float('inf')
        assigned_player = -1

        for player in players:
            if player['cls'] != '2':  # skip non-players
                continue

            player_id = player['id']
            player_bbox = player['bbox']

            # Use bottom left and right corners of bbox
            distance_left = measure_distance((player_bbox[0], player_bbox[3]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[3]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
