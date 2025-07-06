from .inning_simulator import InningSimulator

class Game:
    def __init__(self, team, innings=9):
        self.team = team
        self.innings = innings
        self.inning_simulator = InningSimulator(team)
    
    def simulate_game(self):
        total_runs = 0
        for _ in range(self.innings):
            total_runs += self.inning_simulator.simulate_inning()
        return total_runs

    def reset_game(self, team):
        """チーム（打順）を切り替えて初期化"""
        self.team = team
        self.inning_simulator = InningSimulator(team)