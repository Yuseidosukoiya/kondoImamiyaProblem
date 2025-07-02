import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations

# --- 1. Player クラス ---
class Player:
    def __init__(self, name, stats):
        """
        :param name: 選手名（例: 'imamiya', 'kondo'）
        :param stats: 辞書形式の成績データ（打席、安打、二塁打、…）
        """
        self.name = name
        self.stats = stats
        self.probabilities = self.calc_probabilities()
        
    def calc_probabilities(self):
        singles = self.stats['安打'] - (self.stats['二塁打'] + self.stats['三塁打'] + self.stats['本塁打'])
        p_single = singles / self.stats['打席']
        p_double = self.stats['二塁打'] / self.stats['打席']
        p_triple = self.stats['三塁打'] / self.stats['打席']
        p_hr = self.stats['本塁打'] / self.stats['打席']
        p_walk = (self.stats['四球'] + self.stats['故意四'] + self.stats['死球']) / self.stats['打席']
        p_out = 1 - (p_single + p_double + p_triple + p_hr + p_walk)
        return {
            'single': p_single,
            'double': p_double,
            'triple': p_triple,
            'hr': p_hr,
            'walk': p_walk,
            'out': p_out
        }
    
    def plate_appearance(self):
        outcomes = ['single', 'double', 'triple', 'hr', 'walk', 'out']
        probs = [self.probabilities[outcome] for outcome in outcomes]
        return np.random.choice(outcomes, p=probs)

# --- 2. Team クラス ---
class Team:
    def __init__(self, lineup):
        """
        :param lineup: Player オブジェクトのリスト（打順）
        """
        self.lineup = lineup

# --- 3. InningSimulator クラス ---
class InningSimulator:
    def __init__(self, team):
        self.team = team
        self.batter_index = 0  # 打順はゲーム全体で継続
        self.reset_inning()

    def reset_inning(self):
        self.bases = {'1st': None, '2nd': None, '3rd': None}
        self.outs = 0
        self.runs = 0

    def process_walk(self, batter):
        if self.bases['1st'] is None:
            self.bases['1st'] = batter
        else:
            displaced = self.bases['1st']
            self.bases['1st'] = batter
            if self.bases['2nd'] is None:
                self.bases['2nd'] = displaced
            else:
                displaced2 = self.bases['2nd']
                self.bases['2nd'] = displaced
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = displaced2
                else:
                    self.runs += 1
                    self.bases['3rd'] = displaced2

    def process_single(self, batter):
        if self.bases['3rd'] is not None:
            self.runs += 1
            self.bases['3rd'] = None
        if self.bases['2nd'] is not None:
            if self.bases['2nd'].name == 'imamiya':
                self.runs += 1
            else:
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = self.bases['2nd']
                else:
                    self.runs += 1
            self.bases['2nd'] = None
        if self.bases['1st'] is not None:
            if self.bases['2nd'] is None:
                self.bases['2nd'] = self.bases['1st']
            else:
                if self.bases['3rd'] is None:
                    self.bases['3rd'] = self.bases['1st']
                else:
                    self.runs += 1
            self.bases['1st'] = None
        self.bases['1st'] = batter

    def process_double(self, batter):
        for base in ['3rd', '2nd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        if self.bases['1st'] is not None:
            self.bases['3rd'] = self.bases['1st']
            self.bases['1st'] = None
        self.bases['2nd'] = batter

    def process_triple(self, batter):
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.bases['3rd'] = batter

    def process_hr(self, batter):
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.runs += 1

    def simulate_inning(self):
        self.reset_inning()
        while self.outs < 3:
            batter = self.team.lineup[self.batter_index % len(self.team.lineup)]
            outcome = batter.plate_appearance()
            if outcome == 'out':
                self.outs += 1
            elif outcome == 'walk':
                self.process_walk(batter)
            elif outcome == 'single':
                self.process_single(batter)
            elif outcome == 'double':
                self.process_double(batter)
            elif outcome == 'triple':
                self.process_triple(batter)
            elif outcome == 'hr':
                self.process_hr(batter)
            self.batter_index += 1
        return self.runs

# --- 4. Game クラス ---
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

# --- 5. 選手の成績データと Player オブジェクトの生成 ---
players_stats = {
    'imamiya': {
        '打席': 538,
        '安打': 121,
        '二塁打': 25,
        '三塁打': 4,
        '本塁打': 6,
        '四球': 46,
        '故意四': 2,
        '死球': 3,
    },
    'kondo': {
        '打席': 535,
        '安打': 137,
        '二塁打': 29,
        '三塁打': 2,
        '本塁打': 19,
        '四球': 92,
        '故意四': 11,
        '死球': 6,
    }
}

players_dict = {}
for name, stats in players_stats.items():
    players_dict[name] = Player(name, stats)

# --- 6. ビット全探索によるラインナップシミュレーション ---
def simulate_full_search(num_simulations=10000):
    """
    9打席それぞれをビット（0: 近藤, 1: 今宮）として全探索し、各ラインナップごとの
    9イニングゲームの平均得点を求める。
    """
    results = []
    for lineup_int in range(2**9):  # 0～511
        # 上位ビットが先頭打者となるように9ビットのリストに変換
        pattern = [(lineup_int >> (8 - i)) & 1 for i in range(9)]
        batting_order = [players_dict['imamiya'] if bit == 1 else players_dict['kondo'] for bit in pattern]
        team = Team(batting_order)
        total_runs = 0
        for _ in range(num_simulations):
            game = Game(team, innings=9)
            total_runs += game.simulate_game()
        avg_runs = total_runs / num_simulations
        results.append({
            'lineup_int': lineup_int,
            'pattern': pattern,
            'num_imamiya': sum(pattern),
            'avg_runs': avg_runs,
            'pattern_str': ''.join(str(bit) for bit in pattern)
        })
    return results

# シミュレーション実行（例：各ラインナップあたり10000試行）
results = simulate_full_search(num_simulations=1000)
df = pd.DataFrame(results)

# 今宮の人数順＋ラインナップ番号順にソート
df.sort_values(by=['num_imamiya', 'lineup_int'], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- 7. ヒートマップ用データ作成 ---
# （1）打順パターンのヒートマップ用行列：各行が1ラインナップ、各列が打順位置（1: 今宮, 0: 近藤）
# pattern_matrix = np.array(df['pattern'].tolist())  # shape: (512, 9)

# plt.figure(figsize=(10, 20))
# sns.heatmap(pattern_matrix, cmap="coolwarm", cbar=True, 
#             xticklabels=[f"{i+1}番" for i in range(9)])
# plt.title("バッティングオーダーパターン (1=今宮, 0=近藤)")
# plt.ylabel("ラインナップ（今宮人数でソート）")
# plt.xlabel("打順")
# plt.show()

# （2）平均得点のヒートマップ
# 512ラインナップの平均得点を 16 x 32 の行列にリシェイプして表示
avg_runs_array = np.array(df['avg_runs'])
avg_runs_matrix = avg_runs_array.reshape((16, 32))

# Also reshape the grouping information: number of imamiya per lineup
group_array = np.array(df['num_imamiya'])
group_matrix = group_array.reshape((16, 32))

fig, ax = plt.subplots(figsize=(12, 6))
heatmap = sns.heatmap(avg_runs_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, ax=ax)
ax.set_title("Average Runs per Lineup (9-Inning Game)", fontsize=14)
ax.set_xlabel("Lineup Index (Column)", fontsize=12)
ax.set_ylabel("Lineup Index (Row)", fontsize=12)

rows, cols = group_matrix.shape

# Draw vertical boundaries where the group changes between adjacent columns
for i in range(rows):
    for j in range(cols - 1):
        if group_matrix[i, j] != group_matrix[i, j+1]:
            # Draw a vertical line between column j and j+1 for row i
            # In heatmap coordinates, a cell spans from coordinate j to j+1 horizontally, i to i+1 vertically.
            ax.plot([j+1, j+1], [i, i+1], color='black', lw=2)

# Draw horizontal boundaries where the group changes between adjacent rows
for i in range(rows - 1):
    for j in range(cols):
        if group_matrix[i, j] != group_matrix[i+1, j]:
            # Draw a horizontal line between row i and i+1 for column j
            ax.plot([j, j+1], [i+1, i+1], color='black', lw=2)

plt.tight_layout()
plt.show()

# （3）今宮の人数別の得点分布（箱ひげ図）
# plt.figure(figsize=(8, 6))
# sns.boxplot(x="num_imamiya", y="avg_runs", data=df)
# plt.title("ラインナップ中の今宮の人数別 平均得点")
# plt.xlabel("ラインナップ中の今宮の人数")
# plt.ylabel("平均得点")
# plt.show()

