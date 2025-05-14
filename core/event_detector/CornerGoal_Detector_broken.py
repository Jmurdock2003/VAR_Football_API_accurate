# CornerGoal_Detector.py: Detects corner or goal-kick situations based on ball position
from .Rule_Knowledge_Graph import RuleKnowledgeGraph

class CornerGoalDetector:
    """
    Determines if a ball out-of-bounds event corresponds to a corner
    or a goal kick, using the last team that touched the ball.

    Attributes:
      field_width (int): Width of the pitch in pixels (horizontal length).
      goal_width (int): Approximate width of the goal area in pixels.
      kg: Instance of RuleKnowledgeGraph for rule retrieval (unused here).
    """
    def __init__(self, field_width=1280, goal_width=300):
        # Store pitch dimensions for boundary checks
        self.field_width = int(field_width)
        self.goal_width = int(goal_width)

        # Knowledge graph for soccer rules (rules not applied in this code)
        self.kg = RuleKnowledgeGraph()
        # Retrieve rules for corners and goal kicks
        self.corner_rule = self.kg.get_rule("Corner Rule")
        self.goal_kick_rule = self.kg.get_rule("Goal Kick Rule")

    def check_corner_goal(self, ball_position, last_team_touch):
        """
        Decide whether a ball out event is a corner or goal-kick.

        Args:
          ball_position (tuple[float, float]): (x, y) pixel coordinates of the ball.
          last_team_touch (int): Team ID (1 or 2) that last touched the ball.

        Returns:
          str or None: "corner", "goal_kick", or None if ball still in play.
        """
        # Validate inputs
        if (not isinstance(ball_position, (list, tuple)) or len(ball_position) != 2
            or last_team_touch not in (1, 2)):
            return None

        x, y = ball_position
        # Assume field height ≈ 0.65 * width for vertical boundary check
        # Top boundary (y < 0) and bottom boundary (y > height)
        pitch_height = self.field_width * 0.65

        # Ball out of play vertically
        if y < 0 or y > pitch_height:
            # If attacking team last touched, award goal kick to defense
            if last_team_touch == 1:
                return "goal_kick"
            # If defending team last touched, award corner to attack
            elif last_team_touch == 2:
                return "corner"

        # Ball still in play or not out of bounds
        return None

# BROKEN CODE: This code cannot work properly as the pitch detection is not implemented.
# The pitch detection is needed so you can find when the ball is out of the pitch.
