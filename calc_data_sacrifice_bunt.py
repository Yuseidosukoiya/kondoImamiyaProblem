# calc_data_sacrifice_bunt.py

from scipy.stats import beta  # pip install scipy で入ります

# 経験ベイズ推定による犠打成功率計算（そのまま流用）
def compute_eb_success(players, mu, m0=20):
    """
    players: [{"name": str, "attempt": n_i, "success": s_i}, ...]
    mu: リーグ平均成功率
    m0: 事前疑似試行数（Shrinkageの強さ）
    
    return: [{"name": str, "p_success": float, "eb_success": float}, ...]
    """

    result = []

    for p in players:
        name = p["name"]
        n = p["attempt"]
        s = p["success"]

        # 経験ベイズ推定値（事後平均）
        eb = (s + mu * m0) / (n + m0) if n + m0 > 0 else mu
        
        result.append({
            "name": name,
            "p_success": (s / n) if n > 0 else None,
            "eb_success": eb
        })

    return result


# ベイズ事後分布に基づく保守的推定（分位点）
def compute_posterior_lower(players, mu, m0=20, q=0.25):
    """
    players: [{"name": str, "attempt": n_i, "success": s_i}, ...]
    mu: リーグ全体の平均成功率
    m0: 事前疑似試行数（= α + β に相当）
    q: 下側分位点（例: 0.25 なら 25%点）

    事前分布: p ~ Beta(α, β)
      α + β = m0
      α / (α + β) = μ

    事後分布: p_i | (s_i, n_i) ~ Beta(s_i + α, n_i - s_i + β)

    return: [{
        "name": str,
        "attempt": int,
        "success": int,
        "p_success": float | None,
        "eb_success": float,
        "lower_q": float,       # 事後分布の下側 q 分位点
    }, ...]
    """

    # 事前分布のパラメータ α, β を μ と m0 から決定
    alpha = mu * m0
    beta_prior = (1.0 - mu) * m0

    result = []

    for p in players:
        name = p["name"]
        n = p["attempt"]
        s = p["success"]

        # 経験ベイズ推定値（事後平均）
        eb = (s + alpha) / (n + alpha + beta_prior) if (n + alpha + beta_prior) > 0 else mu

        # 事後分布のパラメータ
        a_post = s + alpha
        b_post = (n - s) + beta_prior

        # Beta(a_post, b_post) の下側 q 分位点
        lower_q = beta.ppf(q, a_post, b_post)

        result.append({
            "name": name,
            "attempt": n,
            "success": s,
            "p_success": (s / n) if n > 0 else None,
            "eb_success": eb,
            "lower_q": lower_q,
        })

    return result


if __name__ == "__main__":
    # 2014年パ・リーグ犠打成功率
    mu_2014 = 1628 / 1879

    #犠打版
    sb_players = [
        {"name": "Y. Honda", "attempt": 17, "success": 16},
        {"name": "A. Nakamura", "attempt": 3, "success": 3},
        {"name": "Y. Yanagita", "attempt": 0, "success": 0},
        {"name": "S. Uchikawa", "attempt": 0,  "success": 0},
        {"name": "Lee Dae-Ho", "attempt": 0,  "success": 0},
        {"name": "Y. Hasegawa", "attempt": 0, "success": 0},
        {"name": "N. Matsuda", "attempt": 2, "success": 1},
        {"name": "S. Tsuruoka", "attempt": 18,  "success": 17},
        {"name": "K. Imamiya", "attempt": 71, "success": 62},
    ]

    # # 経験ベイズ推定値（事後平均）　下の関数に乗ってるので
    # eb_values = compute_eb_success(sb_players, mu_2014, m0=20)
    # print("=== EB mean (posterior expectation) ===")
    # for r in eb_values:
    #     print(r)

    # ベイズ事後分布の25%点（保守的な下限）
    posterior_lower_values = compute_posterior_lower(sb_players, mu_2014, m0=20, q=0.25)
    print("\n=== Posterior 25% quantile (conservative lower bound) ===")
    for r in posterior_lower_values:
        print(r)

