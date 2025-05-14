import sys
# Ensure project root is in Python path for relative imports
sys.path.append('../')  # adjust path as necessary

from utils.bbox_utils import get_centre, measure_distance  # utilities for geometry calculations

class PlayerBallAssigner:
    """
    Assigns the ball to the nearest player based on bounding box proximity.
    Uses the bottom corners of each player's box to approximate kicking contact.
    """
    def __init__(self):
        # Maximum allowed distance (in pixels) between ball and player for assignment
        self.max_player_ball_distance = 70.0

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Determine which player is closest to the ball.

        Arguments:
            players (list of dict): tracked player data with 'cls', 'bbox', 'id'.
            ball_bbox (list of float): [x1, y1, x2, y2] coordinates of the ball.

        Returns:
            int: ID of assigned player, or -1 if none within threshold.
        """
        # Compute the (x, y) center of the ball
        ball_position = get_centre(ball_bbox)

        assigned_player = -1
        minimum_distance = float('inf')  # start with an infinitely large distance

        # Iterate over tracked objects to find eligible players
        for player in players:
            # Only consider class '2' (players), skip referees or ball
            if player.get('cls') != '2':
                continue

            player_id = player.get('id')
            player_bbox = player.get('bbox', [])
            if len(player_bbox) != 4:
                # Skip malformed bounding boxes
                continue

            # Approximate foot contact by using the bottom-left and bottom-right corners
            bottom_left = (player_bbox[0], player_bbox[3])
            bottom_right = (player_bbox[2], player_bbox[3])

            # Compute distances from each corner to the ball center
            dist_left = measure_distance(bottom_left, ball_position)
            dist_right = measure_distance(bottom_right, ball_position)
            distance = min(dist_left, dist_right)

            # Select the player with the smallest valid distance
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player
