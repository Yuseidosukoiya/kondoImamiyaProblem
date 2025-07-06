import pandas as pd
import numpy as np
from models.team import Team
from .game import Game

def simulate_full_search(players_dict, num_simulations=10000):
    """
    9打席それぞれをビット（0: 近藤, 1: 今宮）として全探索し、各ラインナップごとの
    9イニングゲームの平均得点を求める。
    """
    results = []
    game = Game(Team([players_dict['kondo']]*9), innings=9)
    for lineup_int in range(2**9):
        pattern = [(lineup_int >> (8 - i)) & 1 for i in range(9)]
        batting_order = [players_dict['imamiya'] if bit == 1 else players_dict['kondo'] for bit in pattern]
        team = Team(batting_order)
        total_runs = 0
        for _ in range(num_simulations):
            game.reset_game(team)
            total_runs += game.simulate_game()
        avg_runs = total_runs / num_simulations
        results.append({
            'lineup_int': lineup_int,
            'pattern': pattern,
            'num_imamiya': sum(pattern),
            'avg_runs': avg_runs,
            'pattern_str': ''.join(str(bit) for bit in pattern)
        })
    return pd.DataFrame(results)
