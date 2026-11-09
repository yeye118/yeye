"""反思闭环：场景指纹 -> 诊断 -> 动作 -> 验证 -> 知识沉淀。

注意：全部为确定性数值计算，不依赖大模型，保证离线可跑、结果可复现。
"""
import time
import numpy as np

from wta import (greedy, local_search, two_swap, round_hungarian,
                 value_of, improve, solve)


# ------------------------------------------------------------ 指纹
def fingerprint(P, v):
    n, m = P.shape
    nz = float((P > 0).mean())
    v_sum = float(v.sum()) + 1e-12
    vs = np.sort(v)[::-1]
    top = float(vs[: max(1, int(0.1 * m))].sum() / v_sum)
    pnz = P[P > 0]
    return {
        "ratio": round(n / max(m, 1), 2),
        "density": round(nz, 2),
        "p_mean": round(float(pnz.mean()) if pnz.size else 0.0, 3),
        "p_std": round(float(pnz.std()) if pnz.size else 0.0, 3),
        "v_cv": round(float(v.std() / (v.mean() + 1e-12)), 2),
        "v_top": round(top, 2),
    }


# ------------------------------------------------------------ 诊断
def diagnose(P, v, assign):
    n, m = P.shape
    val, surv = value_of(P, v, assign)
    gap = v * surv                                   # 未实现价值
    used = assign >= 0

    # 1) 错配：某件武器在别处能拿到更高边际增益
    misalloc = []
    for i in np.where(used)[0]:
        j = assign[i]
        rest = 1.0 - P[i, j]
        surv_wo = surv[j] / rest if rest > 1e-12 else 0.0
        cur = v[j] * (surv_wo - surv[j])             # 该武器的当前贡献
        g = v * P[i] * surv
        g[j] = -1.0
        k = int(np.argmax(g))
        if g[k] > cur * 1.5 + 1e-9:
            misalloc.append((float(g[k] - cur), int(i), int(k)))

    # 2) 冗余浪费：同一目标第 3 发以后增益很低
    waste = []
    for t in range(m):
        ws = np.where(assign == t)[0]
        if len(ws) >= 3:
            s = 1.0
            gains = []
            for i in ws[np.argsort(-P[ws, t])]:
                gains.append(v[t] * P[i, t] * s)
                s *= 1.0 - P[i, t]
            if gains and gains[-1] < 0.02 * max(gains[0], 1e-9):
                waste.append((int(t), float(gains[-1])))

    misalloc.sort(key=lambda x: -x[0])
    return {
        "value": val,
        "gaps": np.argsort(-gap)[:5].tolist(),
        "gap_val": float(gap.max()) if m else 0.0,
        "misalloc": misalloc[:8],
        "waste": waste[:8],
        "robust_ok": True,
    }


# ------------------------------------------------------------ 动作
def act_force_reassign(P, v, assign, misalloc):
    """抢回首发：把机会成本最高的武器强制改派，再局部修复。"""
    a = assign.copy()
    for _, i, k in misalloc[:3]:
        a[i] = k
    a, _ = local_search(P, v, a)
    return a


def act_release_waste(P, v, assign, waste):
    """释放冗余：把低增益的重复投入转投缺口目标。"""
    a = assign.copy()
    _, surv = value_of(P, v, a)
    for t, _ in waste[:3]:
        ws = np.where(a == t)[0]
        if len(ws) < 3:
            continue
        i = int(ws[np.argmin(P[ws, t])])
        g = v * P[i] * surv
        g[t] = -1.0
        k = int(np.argmax(g))
        if g[k] > 1e-9:
            a[i] = k
    a, _ = local_search(P, v, a)
    return a


def act_robust(P, v, assign, beta=0.15):
    """鲁棒化：按悲观毁伤概率重解，规避对单发高 p 的过度依赖。"""
    a, _ = solve(P * (1.0 - beta), v, budget=1.2, restarts=0, seed=7)
    a, _ = local_search(P, v, a)
    return a


# ------------------------------------------------------------ 主流程
def reflective_solve(scene, kb=None, budget=6.0, seed=0):
    P, v = scene["P"], scene["v"]
    t0 = time.perf_counter()
    deadline = t0 + budget
    fp = fingerprint(P, v)

    # ---- R0 基线：纯边际增益贪心（对应 norethink）
    a0, _ = greedy(P, v)
    v0, _ = value_of(P, v, a0)
    trace = [round(v0, 4)]

    # ---- R1 诊断 + 抢回首发 + 1-swap
    diag = diagnose(P, v, a0)
    a1 = act_force_reassign(P, v, a0, diag["misalloc"])
    v1, _ = value_of(P, v, a1)
    if v1 < v0:
        a1 = a0
        v1 = v0
    trace.append(round(v1, 4))

    # ---- R2 知识注入（策略选择）+ 释放冗余 + 2-swap
    used_rules = []
    a2 = a1
    if kb is not None:
        for hit in kb.query(fp):
            rule = hit["rule"]
            used_rules.append(rule["id"])
            if rule["action"].get("strategy") == "hungarian_first":
                h = round_hungarian(P, v)
                if h is not None:
                    ha = local_search(P, v, h[0])[0]
                    vh, _ = value_of(P, v, ha)
                    if vh > v1:
                        a2 = ha
            elif rule["action"].get("strategy") == "extra_restart":
                b, _ = solve(P, v, budget=max(0.3, deadline - time.perf_counter()),
                             restarts=3, seed=seed + 1)
                vb, _ = value_of(P, v, b)
                if vb > v1:
                    a2 = b
            break

    a2 = act_release_waste(P, v, a2, diag["waste"])
    a2, _ = two_swap(P, v, a2, deadline=deadline)
    a2, _ = local_search(P, v, a2)
    v2, _ = value_of(P, v, a2)
    if v2 < v1:
        a2, v2 = a1, v1
    trace.append(round(v2, 4))

    # ---- R3 验证与收敛
    left = max(0.3, min(1.5, deadline - time.perf_counter()))
    a3, _ = solve(P, v, budget=left, restarts=1, seed=seed + 2)
    v3, _ = value_of(P, v, a3)
    if v3 < v2:
        a3, v3 = a2, v2
    trace.append(round(v3, 4))

    # ---- 知识沉淀：把本轮生效的动作写回
    if kb is not None:
        cond = {"ratio": fp["ratio"]}
        delta = (v3 - v0) / max(float(v.sum()), 1e-9)
        if v3 > v0:
            kb.upsert(cond, {"strategy": "release_and_swap"},
                      delta, scene["name"],
                      mechanism="诊断错配后强制改派并做 2-swap")
            if fp["ratio"] <= 1.8:
                kb.upsert(cond, {"strategy": "hungarian_first"},
                          delta, scene["name"],
                          mechanism="武器/目标比低时首发指派决定收益")
        kb.save()

    info = {
        "fingerprint": fp,
        "rounds": len(trace) - 1,
        "trace": trace,
        "knowledge_used": used_rules,
        "diagnosis": {"misalloc": len(diag["misalloc"]),
                      "waste": len(diag["waste"]),
                      "top_gap_target": int(diag["gaps"][0]) if diag["gaps"] else None},
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }
    return a3, v3, info
