import numpy as np
from itertools import combinations

# 選手データ（例として各種イベント数を辞書で定義）
players = {
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

def calc_probabilities(player):
    # シングル: 安打 - (二塁打+三塁打+本塁打)
    singles = player['安打'] - (player['二塁打'] + player['三塁打'] + player['本塁打'])
    p_single = singles / player['打席']
    p_double = player['二塁打'] / player['打席']
    p_triple = player['三塁打'] / player['打席']
    p_hr = player['本塁打'] / player['打席']
    
    # 出塁に関しては四球+故意四+死球
    p_walk = (player['四球'] + player['故意四'] + player['死球']) / player['打席']
    
    # 残りをアウトとする
    p_out = 1 - (p_single + p_double + p_triple + p_hr + p_walk)
    
    return {
        'single': p_single,
        'double': p_double,
        'triple': p_triple,
        'hr': p_hr,
        'walk': p_walk,
        'out': p_out
    }

# 各選手の確率
prob_imamiya = calc_probabilities(players['imamiya'])
prob_kondo = calc_probabilities(players['kondo'])

#　各結果の関数

# シンプルな1打席のシミュレーション
def simulate_plate_appearance(prob):
    # 結果のリストとその確率を用意
    outcomes = ['single', 'double', 'triple', 'hr', 'walk', 'out']
    probs = [prob[outcome] for outcome in outcomes]
    return np.random.choice(outcomes, p=probs)

# 例：1イニング（3アウトになるまで）のシミュレーション
def simulate_inning(lineup, player_probs):
    runs = 0
    outs = 0
    batter_idx = 0
    # bases: 各塁にいるランナーを管理（キーは '1st', '2nd', '3rd'、値は走者名または None）
    bases = {'1st': None, '2nd': None, '3rd': None}

    # 四球：1塁に打者を入れ、もし埋まっていれば順次強制進塁
    def process_walk(batter):
        nonlocal runs
        if bases['1st'] is None:
            bases['1st'] = batter
        else:
            displaced = bases['1st']
            bases['1st'] = batter
            if bases['2nd'] is None:
                bases['2nd'] = displaced
            else:
                displaced2 = bases['2nd']
                bases['2nd'] = displaced
                if bases['3rd'] is None:
                    bases['3rd'] = displaced2
                else:
                    runs += 1
                    bases['3rd'] = displaced2

    # シングルヒット：3塁→2塁→1塁の順にランナーを進塁させる
    def process_single(batter):
        nonlocal runs
        # 3塁のランナーは常にホームイン
        if bases['3rd'] is not None:
            runs += 1
            bases['3rd'] = None
        # 2塁のランナー：imamiyaなら得点、kondoなら3塁に進む
        if bases['2nd'] is not None:
            if bases['2nd'] == 'imamiya':
                runs += 1
            else:
               # もし3塁が空いていれば進塁、埋まっていれば得点
               bases['3rd'] = bases['2nd']
            bases['2nd'] = None
        # 1塁のランナーは必ず2塁に進む
        if bases['1st'] is not None:
            if bases['2nd'] is None:
                bases['2nd'] = bases['1st']
            else:
                if bases['3rd'] is None:
                    bases['3rd'] = bases['1st']
                else:
                    runs += 1
            bases['1st'] = None
        # 打者は1塁に入る
        bases['1st'] = batter

    # 二塁打：2,3塁のランナーは得点、1塁のランナーは3塁に進む（スペースがなければ得点）
    def process_double(batter):
        nonlocal runs
        for base in ['3rd', '2nd']:
            if bases[base] is not None:
                runs += 1
                bases[base] = None
        if bases['1st'] is not None:
            bases['3rd'] = bases['1st']
            bases['1st'] = None
        bases['2nd'] = batter

    # 三塁打：全ランナー得点、打者は3塁に配置
    def process_triple(batter):
        nonlocal runs
        for base in ['1st', '2nd', '3rd']:
            if bases[base] is not None:
                runs += 1
                bases[base] = None
        bases['3rd'] = batter

    # 本塁打：全ランナー＋打者が得点
    def process_hr(batter):
        nonlocal runs
        for base in ['1st', '2nd', '3rd']:
            if bases[base] is not None:
                runs += 1
                bases[base] = None
        runs += 1

    while outs < 3:
        batter = lineup[batter_idx % len(lineup)]
        outcome = simulate_plate_appearance(player_probs[batter])
        if outcome == 'out':
            outs += 1
        elif outcome == 'walk':
            process_walk(batter)
        elif outcome == 'single':
            process_single(batter)
        elif outcome == 'double':
            process_double(batter)
        elif outcome == 'triple':
            process_triple(batter)
        elif outcome == 'hr':
            process_hr(batter)
        batter_idx += 1

    return runs

def generate_lineups(num_imamiya):
    """
    9人中、指定された人数の imamiya を配置する全パターンのラインナップを生成。
    各ラインナップは、リストの各要素が 'imamiya' または 'kondo' となる長さ9のリストです。
    """
    lineups = []
    # 0～8 のインデックスから imamiya を配置する位置を全組み合わせで求める
    for positions in combinations(range(9), num_imamiya):
        lineup = []
        for i in range(9):
            if i in positions:
                lineup.append('imamiya')
            else:
                lineup.append('kondo')
        lineups.append(lineup)
    return lineups

def simulate_game(lineup, player_probs, innings=9):
    """
    指定されたラインナップと選手確率で、9イニング分のゲームをシミュレーションし、
    得点の合計を返す。
    """
    total_runs = 0
    for _ in range(innings):
        total_runs += simulate_inning(lineup, player_probs)
    return total_runs

def simulate_lineups_all(player_probs, num_simulations=10000):
    """
    imamiya の人数を 0 から 9 まで変化させた全シナリオについて、
    各打順組み合わせの9イニングゲームのシミュレーションを実施し、平均得点を計算する。
    結果は各ラインナップの打順（1番打者～9番打者）と平均得点として出力する。
    """
    for num_imamiya in range(10):
        lineups = generate_lineups(num_imamiya)
        print(f"\n=== imamiya: {num_imamiya} 人, kondo: {9-num_imamiya} 人 のラインナップ ===")
        for lineup in lineups:
            total_runs = 0
            for _ in range(num_simulations):
                total_runs += simulate_game(lineup, player_probs, innings=9)
            avg_runs = total_runs / num_simulations
            formatted_lineup = " ".join(f"{i+1}:{'今宮' if player=='imamiya' else '近藤'}" for i, player in enumerate(lineup))
            print(f"{formatted_lineup}   平均得点: {avg_runs:.3f}")

# 例として、選手の打席確率は事前に calc_probabilities で求めた prob_imamiya, prob_kondo を利用
simulate_lineups_all({'imamiya': prob_imamiya, 'kondo': prob_kondo})