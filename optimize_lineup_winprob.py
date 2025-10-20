# -*- coding: utf-8 -*-
"""
optimize_lineup_winprob.py

2014ホークス9名で「試合勝率」最大の打順を求める。
Stage1: 半イニングの期待得点(ER)をビーム探索で上位候補を抽出
Stage2: 候補をゲーム価値DPで評価（到達集合のみ / 遷移プリコンパイル / 盗塁=論文方式）
Numbaが利用可能なら高速価値反復に切替（fallbackあり）
"""
from __future__ import annotations

# ===== 標準ライブラリ =====
import argparse
import random
import time
from collections import deque
from typing import Dict, List, Tuple

# ===== サードパーティ =====
import numpy as np

try:
    from numba import njit  # type: ignore
except Exception:
    njit = None  # numba未導入でもファイルは読み込めるようにする

# =============================================================================
# 定数・設定
# =============================================================================
# コールドゲーム判定
COLD_FROM_INNING = 5
COLD_DIFF = 8

# ゲーム状態の点差切り取り幅（±12）…必要に応じて16〜20に拡張
MAX_DIFF = 12

# 価値反復
MAX_SWEEPS = 150
TOL = 1e-8
ALPHA = 0.7       # アンダーリラクゼーション（0.5〜0.8 推奨）
EPS_STICK = 1e-4  # 方策スティッキネスの閾値（表の3行動での同値回避）

# 行動数：表=3（swing/bunt/steal）、裏=1（swing固定）
MAX_A = 3
# 分岐上限：swing=最大6（OUT,BB,1B,2B,3B,HR）、bunt=2、steal=2
MAX_BRANCH = 6

# =============================================================================
# データ定義：2014 Hawks (Table 1)
# =============================================================================
Player = Dict[str, float]
players: Dict[str, Player] = {
    "Y. Honda":    {"OUT":0.648, "1B":0.217, "2B":0.032, "3B":0.016, "HR":0.000, "BB":0.087, "SAC":0.941, "SB":0.793},
    "A. Nakamura": {"OUT":0.627, "1B":0.231, "2B":0.035, "3B":0.006, "HR":0.006, "BB":0.095, "SAC":0.800, "SB":0.833},
    "Y. Yanagita": {"OUT":0.593, "1B":0.211, "2B":0.029, "3B":0.007, "HR":0.025, "BB":0.136, "SAC":0.000, "SB":0.846},
    "S. Uchikawa": {"OUT":0.653, "1B":0.199, "2B":0.049, "3B":0.002, "HR":0.034, "BB":0.063, "SAC":0.000, "SB":0.0},
    "Lee Dae-Ho":  {"OUT":0.637, "1B":0.195, "2B":0.048, "3B":0.000, "HR":0.031, "BB":0.090, "SAC":0.000, "SB":0.0},
    "Y. Hasegawa": {"OUT":0.624, "1B":0.193, "2B":0.056, "3B":0.006, "HR":0.011, "BB":0.110, "SAC":0.000, "SB":0.500},
    "N. Matsuda":  {"OUT":0.655, "1B":0.185, "2B":0.048, "3B":0.007, "HR":0.043, "BB":0.062, "SAC":0.500, "SB":0.667},
    "S. Tsuruoka": {"OUT":0.750, "1B":0.167, "2B":0.024, "3B":0.018, "HR":0.000, "BB":0.042, "SAC":0.944, "SB":0.0},
    "K. Imamiya":  {"OUT":0.698, "1B":0.174, "2B":0.044, "3B":0.002, "HR":0.005, "BB":0.077, "SAC":0.873, "SB":0.667},
}

default_lineup = ["Y. Honda","A. Nakamura","Y. Yanagita","S. Uchikawa","Lee Dae-Ho","Y. Hasegawa","N. Matsuda","S. Tsuruoka","K. Imamiya"]
worst_lineup   = ["S. Tsuruoka","Y. Hasegawa","K. Imamiya","Y. Honda","N. Matsuda","S. Uchikawa","Lee Dae-Ho","A. Nakamura","Y. Yanagita"]
opt_lineup_doc = ["A. Nakamura","Y. Yanagita","S. Uchikawa","Lee Dae-Ho","Y. Hasegawa","N. Matsuda","Y. Honda","S. Tsuruoka","K. Imamiya"]

# =============================================================================
# 汎用ユーティリティ
# =============================================================================
def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v

def has_runner(bases: int, base: int) -> bool:
    return (bases >> (base - 1)) & 1 == 1

def set_runner(bases: int, base: int, present: bool) -> int:
    mask = 1 << (base - 1)
    return (bases | mask) if present else (bases & ~mask)

def prev_idx(bidx: int) -> int:
    return (bidx - 1) % 9

# =============================================================================
# 塁状況の遷移
# =============================================================================
def walk_transition(bases: int) -> Tuple[int, int]:
    runs = 1 if (has_runner(bases,1) and has_runner(bases,2) and has_runner(bases,3)) else 0
    nb = 0
    nb = set_runner(nb,3, has_runner(bases,3) or has_runner(bases,2))
    nb = set_runner(nb,2, has_runner(bases,2) or has_runner(bases,1))
    nb = set_runner(nb,1, True)
    return nb, runs

def single_transition(bases: int) -> Tuple[int, int]:
    runs = int(has_runner(bases,2)) + int(has_runner(bases,3))
    nb = 0
    if has_runner(bases,1):
        nb = set_runner(nb,3, True)
    nb = set_runner(nb,1, True)
    return nb, runs

def double_transition(bases: int) -> Tuple[int, int]:
    runs = int(has_runner(bases,1)) + int(has_runner(bases,2)) + int(has_runner(bases,3))
    nb = set_runner(0,2, True)
    return nb, runs

def triple_transition(bases: int) -> Tuple[int, int]:
    runs = int(has_runner(bases,1)) + int(has_runner(bases,2)) + int(has_runner(bases,3))
    nb = set_runner(0,3, True)
    return nb, runs

def hr_transition(bases: int) -> Tuple[int, int]:
    runs = 1 + int(has_runner(bases,1)) + int(has_runner(bases,2)) + int(has_runner(bases,3))
    return 0, runs

def bunt_allowed(bases: int, outs: int) -> bool:
    return (outs <= 1) and has_runner(bases, 1)

def steal_allowed(bases: int, outs: int) -> bool:
    return (outs <= 1) and has_runner(bases, 1) and (not has_runner(bases, 2))

# =============================================================================
# 半イニングER：遷移プリコンパイル & 高速値反復
# =============================================================================
def compile_half_transitions(lineup_players: List[Player]):
    """半イニング内の (swing / bunt / steal) の分岐表を事前作成"""
    SW, BU, ST = {}, {}, {}
    for bidx in range(9):
        batter = lineup_players[bidx]
        for bases in range(8):
            for outs in (0, 1, 2):
                key = (bidx, bases, outs)

                # swing
                dist = []
                p_out = batter["OUT"]
                if p_out > 0:
                    dist.append((p_out, min(outs + 1, 3), bases, 0))
                p_bb = batter["BB"]
                if p_bb > 0:
                    nb, runs = walk_transition(bases)
                    dist.append((p_bb, outs, nb, runs))
                for tag, trans in [("1B", single_transition), ("2B", double_transition),
                                   ("3B", triple_transition), ("HR", hr_transition)]:
                    p = batter[tag]
                    if p > 0:
                        nb, runs = trans(bases)
                        dist.append((p, outs, nb, runs))
                SW[key] = dist

                # bunt
                if bunt_allowed(bases, outs) and batter.get("SAC", 0.0) > 0:
                    sac = batter["SAC"]
                    nb_s = set_runner(bases, 1, False)
                    nb_s = set_runner(nb_s, 2, True)
                    BU[key] = [
                        (sac, min(outs + 1, 3), nb_s, 0),
                        (1 - sac, min(outs + 1, 3), bases, 0),
                    ]
                else:
                    BU[key] = SW[key]

                # steal（論文方式：一塁走者=直前打者のSB）
                if steal_allowed(bases, outs):
                    runner = lineup_players[prev_idx(bidx)]
                    ps = runner.get("SB", 0.0)
                    if ps > 0.0:
                        nb_s = set_runner(bases, 1, False)
                        nb_s = set_runner(nb_s, 2, True)
                        nb_f = set_runner(bases, 1, False)
                        ST[key] = [
                            (ps, outs, nb_s, 0),
                            (1 - ps, min(outs + 1, 3), nb_f, 0),
                        ]
                    else:
                        ST[key] = SW[key]
                else:
                    ST[key] = SW[key]
    return SW, BU, ST

def expected_runs_half_fast(order_names: List[str]) -> float:
    """半イニングの期待得点（攻撃側がswing/bunt/steal最適選択）"""
    lineup = [players[n] for n in order_names]
    SW, BU, ST = compile_half_transitions(lineup)

    V = {(o, b, bi): 0.0 for o in (0, 1, 2, 3) for b in range(8) for bi in range(9)}
    for _ in range(300):
        delta = 0.0
        for outs in (0, 1, 2):
            for bases in range(8):
                for bidx in range(9):
                    nb = (bidx + 1) % 9
                    key = (bidx, bases, outs)

                    def q(dist):
                        s = 0.0
                        for p, no, nbases, runs in dist:
                            if no >= 3:
                                s += p * runs
                            else:
                                s += p * (runs + V[(no, nbases, nb)])
                        return s

                    newv = max(q(SW[key]), q(BU[key]), q(ST[key]))
                    old = V[(outs, bases, bidx)]
                    dv = abs(newv - old)
                    if dv > delta:
                        delta = dv
                    V[(outs, bases, bidx)] = newv
        if delta < 1e-10:
            break
    return V[(0, 0, 0)]

# =============================================================================
# Stage1：ビーム探索で候補を作る
# =============================================================================
def shortlist_by_halfER_beam(
    names: List[str],
    top_k: int = 50,
    beam: int = 30,
    iters: int = 150,
    rand_seeds: int = 20,
) -> List[Tuple[float, List[str]]]:
    seeds = [default_lineup, worst_lineup, opt_lineup_doc]
    for _ in range(rand_seeds):
        perm = names[:]
        random.shuffle(perm)
        seeds.append(perm)

    seen = set()
    beam_list: List[Tuple[float, List[str]]] = []
    for ord0 in seeds:
        t = tuple(ord0)
        if t in seen:
            continue
        seen.add(t)
        er0 = expected_runs_half_fast(ord0)
        beam_list.append((er0, ord0))
    beam_list.sort(reverse=True, key=lambda x: x[0])
    beam_list = beam_list[:beam]

    for _ in range(iters):
        cand: List[Tuple[float, List[str]]] = []
        for er, order in beam_list:
            for __ in range(4):  # 各ビームから4候補
                i, j = random.sample(range(9), 2)
                new = order[:]
                new[i], new[j] = new[j], new[i]
                t = tuple(new)
                if t in seen:
                    continue
                seen.add(t)
                er_new = expected_runs_half_fast(new)
                cand.append((er_new, new))
        beam_list += cand
        beam_list.sort(reverse=True, key=lambda x: x[0])
        beam_list = beam_list[:beam]

    top = beam_list[:top_k]
    top.sort(reverse=True, key=lambda x: x[0])
    return top

# =============================================================================
# Stage2：ゲームDP用の遷移（フル版）
# =============================================================================
def swing_transitions_full(batter: Player, bases: int, outs: int) -> List[Tuple[float, Tuple[int, int, int]]]:
    dist: List[Tuple[float, Tuple[int, int, int]]] = []
    p_out = batter["OUT"]
    if p_out > 0:
        dist.append((p_out, (min(outs + 1, 3), bases, 0)))
    p_bb = batter["BB"]
    if p_bb > 0:
        nb, runs = walk_transition(bases)
        dist.append((p_bb, (outs, nb, runs)))
    for tag, trans in [("1B", single_transition), ("2B", double_transition),
                       ("3B", triple_transition), ("HR", hr_transition)]:
        p = batter[tag]
        if p > 0:
            nb, runs = trans(bases)
            dist.append((p, (outs, nb, runs)))
    return dist

def bunt_transitions_full(batter: Player, bases: int, outs: int) -> List[Tuple[float, Tuple[int, int, int]]]:
    sac = batter.get("SAC", 0.0)
    if not (bunt_allowed(bases, outs) and sac > 0):
        return swing_transitions_full(batter, bases, outs)
    nb_s = set_runner(bases, 1, False)
    nb_s = set_runner(nb_s, 2, True)
    return [(sac, (min(outs + 1, 3), nb_s, 0)), (1 - sac, (min(outs + 1, 3), bases, 0))]

def compile_game_transitions(lineup: List[Player], side_tag: str):
    SW, BU, ST = {}, {}, {}
    for bidx in range(9):
        for bases in range(8):
            for outs in (0, 1, 2):
                key = (side_tag, bidx, bases, outs)
                SW[key] = swing_transitions_full(lineup[bidx], bases, outs)
                BU[key] = bunt_transitions_full(lineup[bidx], bases, outs)
                # steal（論文方式：直前打者のSB）
                if steal_allowed(bases, outs):
                    runner = lineup[prev_idx(bidx)]
                    ps = runner.get("SB", 0.0)
                    if ps > 0:
                        nb_s = set_runner(bases, 1, False)
                        nb_s = set_runner(nb_s, 2, True)
                        nb_f = set_runner(bases, 1, False)
                        ST[key] = [(ps, (outs, nb_s, 0)), (1 - ps, (min(outs + 1, 3), nb_f, 0))]
                    else:
                        ST[key] = [(1.0, (outs, bases, 0))]  # 無効化
                else:
                    ST[key] = [(1.0, (outs, bases, 0))]
    return SW, BU, ST

# =============================================================================
# 到達集合（R）BFS と 終局判定
# =============================================================================
def next_half(inn: int, half: int) -> Tuple[int, int]:
    return (inn, 1) if half == 0 else (inn + 1, 0)

def terminal_value(inn: int, half: int, outs: int, rd: int) -> Tuple[bool, float]:
    # コールド
    if inn >= COLD_FROM_INNING and abs(rd) >= COLD_DIFF:
        return True, 1.0 if rd > 0 else 0.0

    # 9回表終了時点でホームがリードなら試合終了（裏は省略）
    if inn == 9 and half == 0 and outs == 3 and rd < 0:
        return True, 0.0  # ビジター視点で負け

    # 9回裏終了時点
    if inn == 9 and half == 1 and outs == 3:
        return True, 1.0 if rd > 0 else (0.0 if rd < 0 else 0.5)

    return False, 0.0

def reachable_states(SW_A, BU_A, ST_A, SW_H):
    start = (1, 0, 0, 0, 0, 0, 0)
    Q, seen = deque([start]), {start}
    while Q:
        inn, half, outs, bases, bA, bH, rd = Q.popleft()

        # コールド/終局
        term, _ = terminal_value(inn, half, outs, rd)
        if term:
            continue

        # outs>=3 は次ハーフへ（遷移辞書を見ない）
        if outs >= 3:
            ninn, nhalf = next_half(inn, half)
            s2 = (ninn, nhalf, 0, 0, bA, bH, rd)
            if s2 not in seen:
                seen.add(s2)
                Q.append(s2)
            continue

        if half == 0:  # 表：攻撃側（3行動）
            key = ("A", bA, bases, outs)
            next_bA = (bA + 1) % 9
            branches = SW_A.get(key, []) + BU_A.get(key, []) + ST_A.get(key, [])
            for p, (no, nb, runs) in branches:
                if p == 0:
                    continue
                nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                term2, _ = terminal_value(inn, half, no, nrd)
                if term2:
                    continue
                s2 = (inn, half, no, nb, next_bA, bH, nrd)
                if s2 not in seen:
                    seen.add(s2)
                    Q.append(s2)
        else:  # 裏：相手はSwing固定
            key = ("H", bH, bases, outs)
            next_bH = (bH + 1) % 9
            branches = SW_H.get(key, [])
            for p, (no, nb, runs) in branches:
                if p == 0:
                    continue
                nrd = clamp(rd - runs, -MAX_DIFF, MAX_DIFF)
                term2, _ = terminal_value(inn, half, no, nrd)
                if term2:
                    continue
                s2 = (inn, half, no, nb, bA, next_bH, nrd)
                if s2 not in seen:
                    seen.add(s2)
                    Q.append(s2)
    return list(seen)

# =============================================================================
# Stage2：Numba向け配列構築 & 高速価値反復
# =============================================================================
def build_state_index(R: List[Tuple[int, int, int, int, int, int, int]]):
    states = sorted(R)  # 再現性のため順序固定
    sid_of = {s: i for i, s in enumerate(states)}
    S = len(states)
    side_is_away = np.zeros(S, dtype=np.uint8)  # 表=1, 裏=0
    for i, (inn, half, outs, bases, bA, bH, rd) in enumerate(states):
        side_is_away[i] = 1 if half == 0 else 0
    return states, sid_of, side_is_away

def build_numba_arrays(SW_A, BU_A, ST_A, SW_H, states, sid_of):
    S = len(states)
    prob     = np.zeros((S, MAX_A, MAX_BRANCH), dtype=np.float64)
    reward   = np.zeros((S, MAX_A, MAX_BRANCH), dtype=np.float64)
    next_sid = np.full(  (S, MAX_A, MAX_BRANCH), -1, dtype=np.int32)
    branch_n = np.zeros((S, MAX_A), dtype=np.int32)

    for sid, (inn, half, outs, bases, bA, bH, rd) in enumerate(states):
        if outs >= 3:
            # 半イニング終了 → 次ハーフ開始（act=0のみ使用）
            ninn, nhalf = next_half(inn, half)
            ns = (ninn, nhalf, 0, 0, bA, bH, rd)
            term2, tv2 = terminal_value(ninn, nhalf, 0, rd)
            prob[sid, 0, 0] = 1.0
            if term2:
                reward[sid, 0, 0] = tv2
                next_sid[sid, 0, 0] = -1
            else:
                reward[sid, 0, 0] = 0.0
                next_sid[sid, 0, 0] = sid_of.get(ns, -1)
            branch_n[sid, 0] = 1
            continue

        if half == 0:
            key = ("A", bA, bases, outs)
            nbA = (bA + 1) % 9
            # act=0: swing
            b = 0
            for p, (no, nb, runs) in SW_A.get(key, []):
                if p == 0.0:
                    continue
                nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                term2, tv2 = terminal_value(inn, half, no, nrd)
                prob[sid, 0, b] = p
                if term2:
                    reward[sid, 0, b] = tv2
                    next_sid[sid, 0, b] = -1
                else:
                    reward[sid, 0, b] = 0.0
                    next_sid[sid, 0, b] = sid_of.get((inn, half, no, nb, nbA, bH, nrd), -1)
                b += 1
            branch_n[sid, 0] = b

            # act=1: bunt
            b = 0
            for p, (no, nb, runs) in BU_A.get(key, []):
                if p == 0.0:
                    continue
                nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                term2, tv2 = terminal_value(inn, half, no, nrd)
                prob[sid, 1, b] = p
                if term2:
                    reward[sid, 1, b] = tv2
                    next_sid[sid, 1, b] = -1
                else:
                    reward[sid, 1, b] = 0.0
                    next_sid[sid, 1, b] = sid_of.get((inn, half, no, nb, nbA, bH, nrd), -1)
                b += 1
            branch_n[sid, 1] = b

            # act=2: steal（打者は消費しない）
            b = 0
            for p, (no, nb, runs) in ST_A.get(key, []):
                if p == 0.0:
                    continue
                nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                term2, tv2 = terminal_value(inn, half, no, nrd)
                prob[sid, 2, b] = p
                if term2:
                    reward[sid, 2, b] = tv2
                    next_sid[sid, 2, b] = -1
                else:
                    reward[sid, 2, b] = 0.0
                    next_sid[sid, 2, b] = sid_of.get((inn, half, no, nb, bA, bH, nrd), -1)
                b += 1
            branch_n[sid, 2] = b

        else:
            key = ("H", bH, bases, outs)
            nbH = (bH + 1) % 9
            b = 0
            for p, (no, nb, runs) in SW_H.get(key, []):
                if p == 0.0:
                    continue
                nrd = clamp(rd - runs, -MAX_DIFF, MAX_DIFF)
                term2, tv2 = terminal_value(inn, half, no, nrd)
                prob[sid, 0, b] = p
                if term2:
                    reward[sid, 0, b] = tv2
                    next_sid[sid, 0, b] = -1
                else:
                    reward[sid, 0, b] = 0.0
                    next_sid[sid, 0, b] = sid_of.get((inn, half, no, nb, bA, nbH, nrd), -1)
                b += 1
            branch_n[sid, 0] = b
            # act=1,2 は未使用（0のまま）
    return prob, reward, next_sid, branch_n

if njit:
    @njit(cache=True, fastmath=True)  # type: ignore[misc]
    def value_iteration_numba(V, prob, reward, next_sid, branch_n, side_is_away,
                              terminal_mask, terminal_value_arr,
                              start_sid, max_sweeps, tol, alpha, eps_stick):
        S = V.shape[0]
        policy = np.full(S, -1, dtype=np.int8)  # 前回行動

        for sweep in range(max_sweeps):
            delta = 0.0
            for sid in range(S):
                if terminal_mask[sid] == 1:
                    newv = terminal_value_arr[sid]
                else:
                    if side_is_away[sid] == 1:
                        # 表：3行動のmax（スティッキネス）
                        best = -1e300
                        best_a = 0
                        for a in range(3):
                            q = 0.0
                            bn = branch_n[sid, a]
                            for k in range(bn):
                                ns = next_sid[sid, a, k]
                                q += prob[sid, a, k] * (reward[sid, a, k] + (0.0 if ns < 0 else V[ns]))
                            if q > best:
                                best = q
                                best_a = a
                        a_prev = policy[sid]
                        if a_prev >= 0:
                            q_prev = 0.0
                            bn = branch_n[sid, a_prev]
                            for k in range(bn):
                                ns = next_sid[sid, a_prev, k]
                                q_prev += prob[sid, a_prev, k] * (reward[sid, a_prev, k] + (0.0 if ns < 0 else V[ns]))
                            if (best - q_prev) < eps_stick:
                                best = q_prev
                                best_a = a_prev
                        policy[sid] = best_a
                        newv = best
                    else:
                        # 裏：act=0固定
                        q = 0.0
                        bn = branch_n[sid, 0]
                        for k in range(bn):
                            ns = next_sid[sid, 0, k]
                            q += prob[sid, 0, k] * (reward[sid, 0, k] + (0.0 if ns < 0 else V[ns]))
                        newv = q

                old = V[sid]
                V[sid] = (1.0 - alpha) * old + alpha * newv
                diff = V[sid] - old
                if diff < 0:
                    diff = -diff
                if diff > delta:
                    delta = diff

            if delta < tol:
                break
        return V[start_sid]

def evaluate_winprob_numba(away_names: List[str], home_names: List[str],
                           V_warm_np: np.ndarray | None = None, log_prefix: str = ""):
    """NumbaでゲームDPを解く（未導入ならPython版にフォールバック）"""
    away = [players[n] for n in away_names]
    home = [players[n] for n in home_names]

    # 遷移プリコンパイル（Python側）
    SW_A, BU_A, ST_A = compile_game_transitions(away, "A")
    SW_H, _, _ = compile_game_transitions(home, "H")

    # 到達集合
    R = reachable_states(SW_A, BU_A, ST_A, SW_H)
    print(f"{log_prefix}Reachable states: {len(R):,}")

    # 状態の配列化
    states, sid_of, side_is_away = build_state_index(R)
    prob, reward, next_sid, branch_n = build_numba_arrays(SW_A, BU_A, ST_A, SW_H, states, sid_of)

    # 終端マスクと終端値
    S = len(states)
    terminal_mask = np.zeros(S, dtype=np.uint8)
    terminal_value_arr = np.zeros(S, dtype=np.float64)
    for i, (inn, half, outs, bases, bA, bH, rd) in enumerate(states):
        term, tv = terminal_value(inn, half, outs, rd)
        if term:
            terminal_mask[i] = 1
            terminal_value_arr[i] = tv

    # 値関数（ウォームスタート対応）
    if V_warm_np is not None and V_warm_np.shape[0] == S:
        V = V_warm_np.copy()
    else:
        V = np.zeros(S, dtype=np.float64)

    # スタート状態ID
    start_sid = sid_of[(1, 0, 0, 0, 0, 0, 0)]

    # numba未導入なら従来版にフォールバック
    if njit is None:
        print(f"{log_prefix}[WARN] numba not found; falling back to Python DP")
        return evaluate_winprob_fast(away_names, home_names)

    # Numba 価値反復
    wp = value_iteration_numba(V, prob, reward, next_sid, branch_n, side_is_away,
                               terminal_mask, terminal_value_arr,
                               start_sid, MAX_SWEEPS, TOL, ALPHA, EPS_STICK)
    return wp, V  # V は次候補のウォームスタートに使える

# =============================================================================
# メインオーケストレーション
# =============================================================================
def optimize_lineup_by_winprob(
    names: List[str],
    home_ref: List[str],
    top_k: int = 50,
    beam: int = 30,
    iters: int = 150,
    rand_seeds: int = 20,
    top_final: int = 10,
):
    t0 = time.perf_counter()
    shortlist = shortlist_by_halfER_beam(names, top_k=top_k, beam=beam, iters=iters, rand_seeds=rand_seeds)
    t1 = time.perf_counter()
    print(f"[Time] Stage1 (beam={beam}, iters={iters}, top_k={top_k}): {t1 - t0:.3f}s")

    ranked = []
    V_warm = None
    for rank, (er, order) in enumerate(shortlist, 1):
        t_s = time.perf_counter()
        # Python版からNumba版に差し替え済み
        wp, V_warm = evaluate_winprob_numba(order, home_ref, V_warm_np=V_warm, log_prefix=f"[Cand {rank:02d}] ")
        t_e = time.perf_counter()
        ranked.append((wp, er, order, t_e - t_s))
        print(f"[Cand {rank:02d}] WP={wp:.6f}  ER_half={er:.6f}  time={t_e - t_s:.2f}s  lineup={order}")

    ranked.sort(reverse=True, key=lambda x: x[0])
    t2 = time.perf_counter()
    print(f"[Time] Stage2 (K={len(shortlist)}): {t2 - t1:.3f}s")
    print(f"[Time] Total: {t2 - t0:.3f}s")
    return ranked[:top_final]

# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=50, help="Stage2へ回す候補数（ビーム出力）")
    parser.add_argument("--beam", type=int, default=30, help="ビーム幅")
    parser.add_argument("--iters", type=int, default=150, help="ビーム反復回数")
    parser.add_argument("--randseeds", type=int, default=20, help="ランダム初期解の数")
    parser.add_argument("--topfinal", type=int, default=10, help="最終表示件数")
    args = parser.parse_args()

    # 最適化対象（この9名の並び替え）とホーム基準
    names = default_lineup[:]
    home_ref = default_lineup[:]

    result = optimize_lineup_by_winprob(
        names,
        home_ref,
        top_k=args.topk,
        beam=args.beam,
        iters=args.iters,
        rand_seeds=args.randseeds,
        top_final=args.topfinal,
    )

    print("\n=== Best lineups by Win Probability (visitor vs Default home; steal=paper-like) ===")
    for i, (wp, er, order, spent) in enumerate(result, 1):
        print(f"[{i:2d}] WP={wp:.9f}  ER_half={er:.6f}  time={spent:.2f}s  lineup={order}")

# =============================================================================
# [deprecated] 従来のPython版ゲーム価値DP（ウォームスタートdict）
# =============================================================================
def evaluate_winprob_fast(
    away_names: List[str],
    home_names: List[str],
    V_warm: Dict | None = None,
    log_prefix: str = "",
) -> Tuple[float, Dict]:
    away = [players[n] for n in away_names]
    home = [players[n] for n in home_names]
    SW_A, BU_A, ST_A = compile_game_transitions(away, "A")
    SW_H, _, _ = compile_game_transitions(home, "H")
    R = reachable_states(SW_A, BU_A, ST_A, SW_H)

    if V_warm is None:
        V = {s: 0.0 for s in R}
    else:
        V = {s: V_warm.get(s, 0.0) for s in R}

    print(f"{log_prefix}Reachable states: {len(R):,}")

    for sweep in range(1, MAX_SWEEPS + 1):
        delta = 0.0
        for (inn, half, outs, bases, bA, bH, rd) in R:
            term, tv = terminal_value(inn, half, outs, rd)
            if term:
                newv = tv
            elif outs == 3:
                ninn, nhalf = next_half(inn, half)
                s2 = (ninn, nhalf, 0, 0, bA, bH, rd)
                newv = V.get(s2, tv)
            elif half == 0:
                nbA = (bA + 1) % 9
                q0 = q1 = q2 = 0.0
                for p, (no, nb, runs) in SW_A[("A", bA, bases, outs)]:
                    nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                    term2, tv2 = terminal_value(inn, half, no, nrd)
                    q0 += p * (tv2 if term2 else V[(inn, half, no, nb, nbA, bH, nrd)])
                for p, (no, nb, runs) in BU_A[("A", bA, bases, outs)]:
                    nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                    term2, tv2 = terminal_value(inn, half, no, nrd)
                    q1 += p * (tv2 if term2 else V[(inn, half, no, nb, nbA, bH, nrd)])
                for p, (no, nb, runs) in ST_A[("A", bA, bases, outs)]:
                    nrd = clamp(rd + runs, -MAX_DIFF, MAX_DIFF)
                    term2, tv2 = terminal_value(inn, half, no, nrd)
                    q2 += p * (tv2 if term2 else V[(inn, half, no, nb, nbA, bH, nrd)])
                newv = max(q0, q1, q2)
            else:
                nbH = (bH + 1) % 9
                q = 0.0
                for p, (no, nb, runs) in SW_H[("H", bH, bases, outs)]:
                    nrd = clamp(rd - runs, -MAX_DIFF, MAX_DIFF)
                    term2, tv2 = terminal_value(inn, half, no, nrd)
                    q += p * (tv2 if term2 else V[(inn, half, no, nb, bA, nbH, nrd)])
                newv = q

            old = V[(inn, half, outs, bases, bA, bH, rd)]
            dv = abs(newv - old)
            if dv > delta:
                delta = dv
            V[(inn, half, outs, bases, bA, bH, rd)] = newv

        if sweep % 20 == 0 or sweep == 1:
            print(f"{log_prefix}  sweep {sweep:3d}: Δ={delta:.3e}")
        if delta < TOL:
            break

    start = (1, 0, 0, 0, 0, 0, 0)
    return V[start], V  # winprob, warm cache

# エントリポイント
if __name__ == "__main__":
    main()
