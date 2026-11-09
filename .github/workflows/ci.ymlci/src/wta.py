"""WTA 求解内核：解析、贪心、1-swap、2-swap、轮次匹配、多起点。

目标函数：P = sum_j v_j * (1 - prod_{i in S_j}(1 - p_ij))
约束：每件武器最多用一次（sum_j x_ij <= 1）
"""
import json
import os
import time
import numpy as np

P_EPS = 0.999999  # 上界截断，避免 p=1 导致除零；对结果影响 1e-6 量级


# ---------------------------------------------------------------- 解析
def parse_scene(path):
    """读取场景，返回 {P, v, n, m, name}。自动适配 P 的方向。"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "P" not in d or "v" not in d:
        raise ValueError("缺少 P 或 v 字段")
    P = np.asarray(d["P"], dtype=float)
    v = np.asarray(d["v"], dtype=float).ravel()
    if P.ndim != 2:
        raise ValueError(f"P 必须是二维数组，实际 {P.shape}")
    if P.shape[1] != len(v):
        if P.shape[0] == len(v):
            P = P.T                      # 行不是武器时转置
        else:
            raise ValueError(f"维度无法自洽: P{P.shape}, len(v)={len(v)}")
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.clip(P, 0.0, P_EPS)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, None)
    return {"P": P, "v": v, "n": P.shape[0], "m": P.shape[1],
            "name": os.path.basename(path)}


# ---------------------------------------------------------------- 目标值
def value_of(P, v, assign):
    surv = np.ones(P.shape[1])
    for i in np.where(assign >= 0)[0]:
        surv[assign[i]] *= 1.0 - P[i, assign[i]]
    return float(np.dot(v, 1.0 - surv)), surv


def target_surv(P, assign, t):
    """单独重算某目标的存活概率（用于交换后精确更新）。"""
    s = 1.0
    for i in np.where(assign == t)[0]:
        s *= 1.0 - P[i, t]
    return s


# ---------------------------------------------------------------- 贪心
def greedy(P, v, rng=None, noise=0.0):
    """边际增益贪心，O(n*m)。noise>0 时做随机扰动（多起点用）。"""
    n, m = P.shape
    assign = np.full(n, -1, dtype=int)
    surv = np.ones(m)
    used = np.zeros(n, bool)
    for _ in range(n):
        G = P * (v * surv)[None, :]
        if noise > 0.0 and rng is not None:
            G = G * (1.0 + noise * rng.standard_normal((n, m)))
        G[used] = -1.0
        k = int(np.argmax(G))
        i, j = divmod(k, m)
        if G[i, j] <= 1e-12:
            break
        assign[i] = j
        used[i] = True
        surv[j] *= 1.0 - P[i, j]
    return assign, surv


# ---------------------------------------------------------------- 1-swap
def local_search(P, v, assign, max_pass=40):
    n, m = P.shape
    a = assign.copy()
    _, surv = value_of(P, v, a)
    for _ in range(max_pass):
        improved = False
        for i in range(n):
            j = a[i]
            if j < 0:
                g = v * P[i] * surv
                k = int(np.argmax(g))
                if g[k] > 1e-12:
                    a[i] = k
                    surv[k] *= 1.0 - P[i, k]
                    improved = True
                continue
            rest = 1.0 - P[i, j]
            surv_wo = surv[j] / rest if rest > 1e-12 else 0.0
            loss = v[j] * (surv_wo - surv[j])       # >=0
            g = v * P[i] * surv
            g[j] = -1.0
            k = int(np.argmax(g))
            if g[k] > loss + 1e-12:
                surv[j] = surv_wo
                a[i] = k
                surv[k] *= 1.0 - P[i, k]
                improved = True
        if not improved:
            break
    return a, surv


# ---------------------------------------------------------------- 2-swap
def two_swap(P, v, assign, max_iter=25, deadline=None):
    """交换两件武器的目标，精确评估，向量化一次评估全部 O(n^2) 对。"""
    n, m = P.shape
    a = assign.copy()
    _, surv = value_of(P, v, a)
    for _ in range(max_iter):
        if deadline is not None and time.perf_counter() > deadline:
            break
        idx = np.where(a >= 0)[0]
        if len(idx) < 2:
            break
        tgt = a[idx]
        A = 1.0 - P[idx, tgt]                    # 各武器的存活因子
        S = surv[tgt] / A                        # 移除该武器后的存活概率
        c1 = v[tgt] * S
        Qa = P[:, tgt]                           # Qa[k, i] = p[k, a[i]]
        Qt = Qa.T[:, idx]                        # Qt[i,k] = p[k, a[i]]
        Qs = Qa[idx, :]                          # Qs[i,k] = p[i, a[k]]
        T1 = c1 * (A - 1.0)
        delta = (T1[:, None] + c1[:, None] * Qt
                 + T1[None, :] + Qs * c1[None, :])
        same = tgt[:, None] == tgt[None, :]
        delta[same] = -1.0
        k = int(np.argmax(delta))
        if delta.flat[k] <= 1e-10:
            break
        ii, kk = divmod(k, len(idx))
        i, j = idx[ii], idx[kk]
        a[i], a[j] = a[j], a[i]
        surv[a[i]] = target_surv(P, a, a[i])
        surv[a[j]] = target_surv(P, a, a[j])
    return a, surv


# ---------------------------------------------------------------- 轮次匹配
def round_hungarian(P, v, rounds=3):
    """每轮每目标至多配发一件，整轮做全局最优匹配；适合武器/目标比低的情形。"""
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return None
    n, m = P.shape
    assign = np.full(n, -1, dtype=int)
    surv = np.ones(m)
    used = np.zeros(n, bool)
    for _ in range(rounds):
        if used.all():
            break
        cand = np.where(~used)[0]
        M = P[np.ix_(cand, np.arange(m))] * (v * surv)[None, :]
        r, c = linear_sum_assignment(-M)
        ok = M[r, c] > 1e-12
        if not ok.any():
            break
        for rr, cc in zip(r[ok], c[ok]):
            i = cand[rr]
            assign[i] = cc
            used[i] = True
            surv[cc] *= 1.0 - P[i, cc]
    return assign, surv


# ---------------------------------------------------------------- 组合
def improve(P, v, assign, deadline=None):
    a, _ = local_search(P, v, assign)
    if deadline is not None and time.perf_counter() > deadline:
        return a
    a, _ = two_swap(P, v, a, deadline=deadline)
    a, _ = local_search(P, v, a)
    return a


def solve(P, v, budget=6.0, restarts=1, seed=0):
    """多策略取优：贪心 / 轮次匹配 / 随机化重起点，再统一精修。"""
    t0 = time.perf_counter()
    deadline = t0 + budget
    n, m = P.shape
    cands = []

    a0, _ = greedy(P, v)
    cands.append(a0)

    ratio = n / max(m, 1)
    if ratio <= 1.8:                       # 稀缺场景：首发指派决定性更强
        h = round_hungarian(P, v)
        if h is not None:
            cands.append(h[0])

    rng = np.random.default_rng(seed)
    for _ in range(max(0, restarts)):
        ar, _ = greedy(P, v, rng=rng, noise=0.15)
        cands.append(ar)

    best_a, best_val = None, -1.0
    for a in cands:
        if time.perf_counter() > deadline:
            break
        a = improve(P, v, a, deadline=deadline)
        val, _ = value_of(P, v, a)
        if val > best_val:
            best_val, best_a = val, a
    if best_a is None:
        best_a = a0
    return best_a, best_val
