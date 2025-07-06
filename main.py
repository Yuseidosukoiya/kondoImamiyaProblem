from models.player import Player
from data.player_stats import players_stats
from simulation.lineup_simulation import simulate_full_search
from viz.visualize import plot_heatmap

# --- Player オブジェクト作成 ---
players_dict = {name: Player(name, stats) for name, stats in players_stats.items()}

# --- シミュレーション実行 ---
df = simulate_full_search(players_dict, num_simulations=10)

# --- ソート ---
df.sort_values(by=['num_imamiya', 'lineup_int'], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- ヒートマップ表示 ---
plot_heatmap(df)
