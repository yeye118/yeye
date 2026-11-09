"""入口：扫描输入目录，逐个场景求解，输出 norethink / rethink 两套结果。

用法：
  python3 src/main.py --task all --test --out-dir /output
  python3 src/main.py --input-dir ./input --out-dir ./output      # 本地自测

输出目录结构（与题目要求一致）：
  <out>/norethink/<与输入同名>.json     关闭反思
  <out>/rethink/<与输入同名>.json       开启反思
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wta import parse_scene, greedy, solve, value_of            # noqa: E402
from reflect import reflective_solve, fingerprint               # noqa: E402
from kb import KnowledgeBase                                     # noqa: E402

TOTAL_BUDGET_S = 540.0     # 整体保险（评测上限 600 s）


def build_payload(scene, assign, mode, extra=None):
    P, v = scene["P"], scene["v"]
    val, _ = value_of(P, v, assign)
    idx = [int(i) for i in range(len(assign)) if assign[i] >= 0]
    pairs = [[int(i), int(assign[i])] for i in idx]
    payload = {
        "scene": os.path.splitext(scene["name"])[0],
        "mode": mode,
        "n_weapons": int(P.shape[0]),
        "m_targets": int(P.shape[1]),
        "assignments": [{"weapon": i, "target": t} for i, t in pairs],
        "pairs": pairs,
        "x": [int(t) for t in assign],
        "expected_value": round(val, 4),
        "total_value": round(float(v.sum()), 4),
        "ratio": round(val / max(float(v.sum()), 1e-9), 6),
    }
    if extra:
        payload["reflection"] = extra
    return payload


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    ap.add_argument("--input-dir", default=os.environ.get("DATA_ROOT", "/input"))
    args, _ = ap.parse_known_args()          # 容忍额外参数

    inp = args.input_dir
    out = args.out_dir
    files = sorted(glob.glob(os.path.join(inp, "*.json")))
    print(f"[main] input={inp} files={len(files)} out={out}", flush=True)

    d_nore = os.path.join(out, "norethink")
    d_re = os.path.join(out, "rethink")
    os.makedirs(d_nore, exist_ok=True)
    os.makedirs(d_re, exist_ok=True)

    kb_path = "/tmp/wta_kb.jsonl" if os.access("/tmp", os.W_OK) else os.path.join(out, "kb.jsonl")
    kb = KnowledgeBase(kb_path)
    t_start = time.perf_counter()

    for k, path in enumerate(files):
        name = os.path.basename(path)
        try:
            scene = parse_scene(path)
        except Exception as e:
            print(f"[warn] 解析失败 {name}: {e}", flush=True)
            continue

        left = TOTAL_BUDGET_S - (time.perf_counter() - t_start)
        budget = max(1.0, min(8.0, left / max(1, len(files) - k)))

        # ---------------- norethink：纯贪心（基线）
        try:
            a0, _ = greedy(scene["P"], scene["v"])
        except Exception:
            traceback.print_exc()
            a0 = -1 * __import__("numpy").ones(scene["n"], dtype=int)
        write_json(os.path.join(d_nore, name), build_payload(scene, a0, "norethink"))

        # ---------------- rethink：反思闭环
        info = None
        try:
            a1, _, info = reflective_solve(scene, kb=kb, budget=budget, seed=k)
        except Exception:
            traceback.print_exc()
            try:
                a1, _ = solve(scene["P"], scene["v"], budget=budget, seed=k)
            except Exception:
                a1 = a0
        write_json(os.path.join(d_re, name), build_payload(scene, a1, "rethink", info))

        p0 = build_payload(scene, a0, "norethink")["ratio"]
        p1 = build_payload(scene, a1, "rethink")["ratio"]
        print(f"[{k+1}/{len(files)}] {name:26s} P1={p0:.4f} P2={p1:.4f} "
              f"Δ={100*(p1-p0):+.2f}pt  {info['elapsed_s'] if info else 0:.2f}s", flush=True)

    print(f"[main] done in {time.perf_counter()-t_start:.1f}s, kb={len(kb)}", flush=True)


if __name__ == "__main__":
    main()
