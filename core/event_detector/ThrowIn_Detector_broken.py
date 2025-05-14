# ThrowIn_Detector.py: Detects which team is awarded a throw-in based on ball exiting pitch boundaries
from .Rule_Knowledge_Graph import RuleKnowledgeGraph

class ThrowInDetector:
    """
    Determines throw-in possession using the ball's out-of-bounds position.

    Parameters:
      field_width (int): Width of the pitch in pixels for boundary checks.
    """
    def __init__(self, field_width=1280):
        # Width of playing area in pixels; if pitch detection not available,
        # this must be set manually after knowing the video resolution or
        # extracted from the pitch detector output (bbox of pitch corners).
        self.field_width = int(field_width)
        # Knowledge graph for throw-in rules (not yet integrated)
        self.kg = RuleKnowledgeGraph()
        # Retrieve the specific rule logic; currently unused
        self.rule = self.kg.get_rule("Throw-In Rule")

    def check_throw_in(self, ball_position, last_team_touch):
        """
        Check if the ball is out of bounds horizontally and assign throw-in.

        Args:
          ball_position (tuple[float, float]): (x, y) coordinates of ball in frame.
          last_team_touch (int): ID of team (1 or 2) that last touched the ball.

        Returns:
          int or None: Team number awarded the throw-in, None if still in bounds.
        """
        # Validate input format
        if (not isinstance(ball_position, (list, tuple)) or
            len(ball_position) != 2 or
            last_team_touch not in (1, 2)):
            # Invalid arguments; cannot determine throw-in
            return None

        x, y = ball_position

        # Ball out on left side (x < 0)
        if x < 0:
            # Award to opposite team of last touch
            return 2 if last_team_touch == 1 else 1

        # Ball out on right side (x > field_width)
        if x > self.field_width:
            # Award to opposite team of last touch
            return 2 if last_team_touch == 1 else 1

        # Ball still in play
        return None

# BROKEN CODE:This code cannot work properly as the pitch detection is not implemented.
# The pitch detection is needed so you can find when the ball is out of the pitch.
