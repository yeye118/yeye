"""轻量知识库：条件-动作-效果的 JSONL 存储，支持增量更新、置信度更新与冲突消解。"""
import hashlib
import json
import os


def _key(cond, action):
    s = json.dumps(cond, sort_keys=True, ensure_ascii=False) + "|" + json.dumps(
        action, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


class KnowledgeBase:
    def __init__(self, path=None):
        self.path = path
        self.items = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            it = json.loads(line)
                            self.items[it["id"]] = it
            except Exception:
                self.items = {}

    # ------------------------------------------------------------ 检索
    def query(self, fp):
        """按场景指纹返回可用规则，相似度加权排序。"""
        out = []
        for it in self.items.values():
            if it.get("deprecated"):
                continue
            sim = _similarity(fp, it.get("condition", {}))
            if sim >= 0.6:
                out.append((sim * it.get("confidence", 0.5), sim, it))
        out.sort(key=lambda x: -x[0])
        return [{"rule": it, "sim": round(sim, 3)} for _, sim, it in out[:5]]

    # ------------------------------------------------------------ 写入
    def upsert(self, cond, action, delta, scene, mechanism=""):
        kid = _key(cond, action)
        it = self.items.get(kid)
        if it is None:
            it = {"id": kid, "condition": cond, "action": action,
                  "effect": {"delta": 0.0, "n": 0}, "confidence": 0.5,
                  "evidence": [], "mechanism": mechanism, "deprecated": False}
            self.items[kid] = it
        e = it["effect"]
        e["delta"] = (e["delta"] * e["n"] + delta) / (e["n"] + 1)
        e["n"] += 1
        # 置信度：Beta 式后验，成功加分、失败减分
        it["confidence"] = min(0.99, max(0.05, it["confidence"] + (0.08 if delta > 0 else -0.08)))
        if scene not in it["evidence"]:
            it["evidence"].append(scene)
        # 冲突消解：同一动作在相似条件下效果反号且证据足够 -> 降权并标记待特化
        if e["n"] >= 3 and e["delta"] < 0:
            it["confidence"] = max(0.05, it["confidence"] - 0.1)
        if it["confidence"] < 0.2:
            it["deprecated"] = True
        return it

    def save(self):
        if not self.path:
            return
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for it in self.items.values():
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.items)


def _similarity(fp, cond):
    """指纹与规则条件的匹配度：条件里列出的字段逐项比较，全部命中才算高相似。"""
    if not cond:
        return 1.0
    hit = 0
    tot = 0
    for k, want in cond.items():
        if k not in fp:
            continue
        tot += 1
        got = fp[k]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if want == 0:
                hit += 1 if abs(got) < 1e-9 else 0
            else:
                hit += 1 if abs(got - want) / abs(want) <= 0.25 else 0
        else:
            hit += 1 if got == want else 0
    if tot == 0:
        return 1.0
    return hit / tot
