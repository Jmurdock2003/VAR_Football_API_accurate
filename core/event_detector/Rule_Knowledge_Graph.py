class RuleKnowledgeGraph:
    def __init__(self):
        self.graph = {
            "Offside": {
                "Check Moment": "Ball is kicked",
                "Conditions": [
                    "Attacker is in opponent's half",
                    "Attacker is ahead of the second-last defender",
                    "Attacker is ahead of the ball"
                ]
            },
            "Throw-In": {
                "Check Moment": "Ball crosses sideline",
                "Conditions": [
                    "Ball completely out",
                    "Crosses left or right boundary"
                ]
            },
            "Goal Kick": {
                "Check Moment": "Ball crosses goal line",
                "Conditions": [
                    "Last touched by attacking team"
                ]
            },
            "Corner": {
                "Check Moment": "Ball crosses goal line",
                "Conditions": [
                    "Last touched by defending team"
                ]
            }
        }

    def get_rule(self, rule_name):
        return self.graph.get(rule_name, None)

    def get_conditions(self, rule_name):
        rule = self.get_rule(rule_name)
        return rule["Conditions"] if rule else []
