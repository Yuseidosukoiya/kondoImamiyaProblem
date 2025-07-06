import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from itertools import combinations

# ----- 既存のシミュレーションコード（Player, Team, InningSimulator, Game, generate_lineups, etc.） -----

# 1. Player クラス
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
        """
        打席数に対して、シングル、ダブル、三塁打、本塁打、四球など各アウトカムの確率を計算する
        """
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
        """
        選手の打席をシミュレーションし、結果を返す。
        """
        outcomes = ['single', 'double', 'triple', 'hr', 'walk', 'out']
        probs = [self.probabilities[outcome] for outcome in outcomes]
        return np.random.choice(outcomes, p=probs)
        
# 2. Team クラス
class Team:
    def __init__(self, lineup):
        """
        :param lineup: Player オブジェクトのリスト（打順）
        """
        self.lineup = lineup

# 3. InningSimulator クラス
class InningSimulator:
    def __init__(self, team):
        """
        :param team: チーム（Team クラスのインスタンス）
        """
        self.team = team
        self.batter_index = 0  # 打順の継続はゲーム全体で管理するため、イニング間でリセットしない
        self.reset_inning()

    def reset_inning(self):
        """イニング開始時に、塁上状況・アウト・得点を初期化"""
        self.bases = {'1st': None, '2nd': None, '3rd': None}
        self.outs = 0
        self.runs = 0

    def process_walk(self, batter):
        """四球の場合の処理（塁が満塁の場合は強制進塁・得点）"""
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
        """シングルヒット：各塁の走者の進塁を処理する。
           ※例として、2塁走者が 'imamiya' なら得点、'kondo' なら3塁進塁とする（シミュレーション上のルール）"""
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
        """二塁打：2,3塁のランナーは得点、1塁の走者は3塁へ進む"""
        for base in ['3rd', '2nd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        if self.bases['1st'] is not None:
            self.bases['3rd'] = self.bases['1st']
            self.bases['1st'] = None
        self.bases['2nd'] = batter

    def process_triple(self, batter):
        """三塁打：全ランナー得点、打者は3塁に配置"""
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.bases['3rd'] = batter

    def process_hr(self, batter):
        """本塁打：全ランナーと打者が得点"""
        for base in ['1st', '2nd', '3rd']:
            if self.bases[base] is not None:
                self.runs += 1
                self.bases[base] = None
        self.runs += 1

    def simulate_inning(self):
        """3アウトになるまでの打席をシミュレーションし、イニングの得点を返す"""
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

# 4. Game クラス
class Game:
    def __init__(self, team, innings=9):
        """
        :param team: Team クラスのインスタンス
        :param innings: ゲームイニング数（デフォルトは9）
        """
        self.team = team
        self.innings = innings
        self.inning_simulator = InningSimulator(team)
    
    def simulate_game(self):
        """全イニングをシミュレーションし、ゲームの合計得点を返す"""
        total_runs = 0
        for _ in range(self.innings):
            total_runs += self.inning_simulator.simulate_inning()
        return total_runs

# 5. 補助関数：ラインナップ生成
def generate_lineups(num_imamiya):
    """
    9人中、指定された人数の imamiya を配置する全パターンのラインナップを生成する。
    各ラインナップは、Player オブジェクトのリストで構成される。
    """
    lineups = []
    # imamiya を配置する位置の組み合わせを算出
    for positions in combinations(range(9), num_imamiya):
        lineup = []
        for i in range(9):
            if i in positions:
                lineup.append(players_dict['imamiya'])
            else:
                lineup.append(players_dict['kondo'])
        lineups.append(lineup)
    return lineups

# 選手の成績データ
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

# Player オブジェクトの生成
players_dict = {}
for name, stats in players_stats.items():
    players_dict[name] = Player(name, stats)

# ----- ここから全体のシミュレーション結果をテーブル＆ヒートマップとして出力する処理 -----

def simulate_lineups_all_heatmap(num_simulations=10000):
    """
    各グループ（imamiya の人数 0～9）の全ラインナップについて
    9イニングゲームのシミュレーションを実施し、各ラインナップの平均得点を記録。
    その結果をテーブルとして出力し、かつヒートマップとして可視化する。
    """
    results = {}  # キー：imamiya の人数、値：各ラインナップの平均得点リスト
    for num_imamiya in range(10):
        lineups = generate_lineups(num_imamiya)
        avg_runs_list = []
        for lineup in lineups:
            team = Team(lineup)
            total_runs = 0
            for _ in range(num_simulations):
                game = Game(team, innings=9)
                total_runs += game.simulate_game()
            avg_runs = total_runs / num_simulations
            avg_runs_list.append(avg_runs)
        results[num_imamiya] = avg_runs_list
        print(f"imamiya: {num_imamiya} 人, ラインナップ数: {len(avg_runs_list)}, 平均得点の平均: {np.mean(avg_runs_list):.3f}")
    
    # 最大列数（= 各グループで最も多いラインナップ数）を求める
    max_lineups = max(len(lst) for lst in results.values())
    
    # 各グループごとに、足りない分は NaN で埋めて 2 次元の配列（リストのリスト）を作成する
    data = []
    for num_imamiya in range(10):
        row = results[num_imamiya]
        if len(row) < max_lineups:
            row = row + [np.nan]*(max_lineups - len(row))
        data.append(row)
    
    # DataFrame 化して結果を確認（各行＝imamiya の人数、各列＝該当グループ内のラインナップ番号）
    df = pd.DataFrame(data, index=[f"imamiya数={i}" for i in range(10)])
    print("\nシミュレーション結果のテーブル（各ラインナップの平均得点）:")
    print(df)
    
    # ヒートマップの作成
    plt.figure(figsize=(12, 6))
    # imshow は自動的に NaN 部分を無視したカラーマッピングを行う
    heatmap = plt.imshow(df, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(heatmap, label='平均得点')
    plt.xlabel('ラインナップ インデックス')
    plt.ylabel('imamiya の人数')
    plt.title('シミュレーション結果 ヒートマップ')
    plt.yticks(ticks=np.arange(10), labels=[f"{i}" for i in range(10)])
    plt.show()

# シミュレーション実行（num_simulations の値は必要に応じて変更してください）
simulate_lineups_all_heatmap(num_simulations=10000)
