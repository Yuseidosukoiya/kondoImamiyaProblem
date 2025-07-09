from models.player import Player
from data.player_stats import players_stats
from simulation.lineup_simulation import simulate_full_search
from viz.visualize import plot_heatmap

# --- Player オブジェクト作成 ---
players_dict = {name: Player(name, stats) for name, stats in players_stats.items()}

# --- シミュレーション実行 ---
df = simulate_full_search(players_dict, num_simulations=10000)

# --- ソート ---
df.sort_values(by=['num_imamiya', 'lineup_int'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 1. 全ラインナップ結果CSV出力
df.to_csv('lineup_results.csv', index=False)

# 2. 今宮人数ごと最高期待値ラインナップ
best_lineups = df.loc[df.groupby('num_imamiya')['avg_runs'].idxmax()]
print("=== 今宮人数ごと最高得点期待値の打順パターン ===")
for _, row in best_lineups.iterrows():
    num_imamiya = row['num_imamiya']
    avg_run = row['avg_runs']
    pattern = row['pattern']
    pattern_readable = [("今宮" if b else "近藤") for b in pattern]
    order_str = " ".join([f"{i+1}番:{name}" for i, name in enumerate(pattern_readable)])
    print(f"今宮{num_imamiya}人: 期待値={avg_run:.3f} {order_str}")

# --- ヒートマップ表示 ---
plot_heatmap(df)