import networkx as nx

class RuleKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        # Rule nodes
        self.graph.add_node("Offside")
        self.graph.add_node("Throw-In")
        self.graph.add_node("Goal Kick")
        self.graph.add_node("Corner")

        # Offside Conditions
        self.graph.add_edge("Offside", "Ball is played or touched by teammate")
        self.graph.add_edge("Offside", "Player is in opponent's half")
        self.graph.add_edge("Offside", "Player is ahead of the ball")
        self.graph.add_edge("Offside", "Player is ahead of the second-last defender")
        self.graph.add_edge("Offside", "Player interferes with play")

        # Throw-In Conditions
        self.graph.add_edge("Throw-In", "Ball crosses touchline")
        self.graph.add_edge("Throw-In", "Ball completely out of play")
        self.graph.add_edge("Throw-In", "Last touched by opponent")

        # Goal Kick Conditions
        self.graph.add_edge("Goal Kick", "Ball crosses goal line")
        self.graph.add_edge("Goal Kick", "Ball not between goalposts")
        self.graph.add_edge("Goal Kick", "Last touched by attacking team")

        # Corner Kick Conditions
        self.graph.add_edge("Corner", "Ball crosses goal line")
        self.graph.add_edge("Corner", "Ball not between goalposts")
        self.graph.add_edge("Corner", "Last touched by defending team")

    def get_conditions(self, rule_name):
        return list(self.graph.successors(rule_name)) if rule_name in self.graph.nodes else []

    def visualize(self, filename="rule_knowledge_graph.png"):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=9, arrows=True)
        plt.savefig(filename)
