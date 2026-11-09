# 反思式决策智能体（提交包）

## 1. 目录

```
solution/
├── Dockerfile
├── run.sh                 # 固定入口，勿改名
├── requirements.txt
├── src/
│   ├── main.py            # 扫描 /input → 求解 → 写 /output
│   ├── wta.py             # 解析 + 贪心 + 1-swap + 2-swap + 轮次匹配 + 多起点
│   ├── reflect.py         # 反思闭环：指纹 → 诊断 → 动作 → 验证 → 沉淀
│   └── kb.py              # 知识库：JSONL，增量更新 / 置信度 / 冲突降权
└── README.md
```

## 2. 运行

```bash
# 评测环境（默认）
bash run.sh --task all --test --out-dir /output

# 本地自测
python3 src/main.py --input-dir ./input --out-dir ./output
```

输入 `/input/*.json`，输出：

```
/output/norethink/<与输入同名>.json   关闭反思（基线）
/output/rethink/<与输入同名>.json     开启反思
```

## 3. 输出格式

字段冗余设计，便于评测端任选其一解析；三者内容严格一致。

```json
{
  "scene": "scene1_0_50_20",
  "mode": "rethink",
  "n_weapons": 50,
  "m_targets": 20,
  "assignments": [{"weapon": 0, "target": 19}, {"weapon": 1, "target": 11}],
  "pairs": [[0, 19], [1, 11]],
  "x": [19, 11, -1, "..."],
  "expected_value": 939.6,
  "total_value": 1036.0,
  "ratio": 0.9069,
  "reflection": {
    "fingerprint": {"ratio": 2.5, "density": 0.53, "p_mean": 0.218},
    "rounds": 3,
    "trace": [926.0, 938.2, 939.6],
    "knowledge_used": ["a1b2c3d4e5"],
    "diagnosis": {"misalloc": 6, "waste": 0, "top_gap_target": 7},
    "elapsed_s": 0.03
  }
}
```

- `x`：长度 `n_weapons`，第 i 位为该武器打的目标下标，`-1` 表示不使用。
- 武器下标 `0..n-1`，目标下标 `0..m-1`，均从 0 开始。
- 每件武器至多出现一次。

## 4. 维度自适应

题面定义 `P` 为 `n×m`（行=武器、列=目标）、`V` 为 `1×m`。程序按 `P.shape[1] == len(v)` 判定；不等且 `P.shape[0] == len(v)` 时自动转置，两向都不自洽则跳过该文件（不写非法结果）。

## 5. 关键设计

| 层 | 做法 | 耗时（500×200） |
|---|---|---|
| 基线 | 边际增益贪心，O(n·m) | 0.11 s |
| 精修 | 1-swap 精确增量的局部搜索 | 0.02 s |
| 精修 | 2-swap 全对向量化评估 | 0.1~0.2 s |
| 换策略 | 武器/目标比 ≤1.8 时加轮次全局匹配 | 0.04 s |
| 反思 | 诊断 → 强制改派 → 释放冗余 → 多起点验证 | 3 轮，≤1.5 s |

无外网、无大模型依赖，结果确定（固定随机种子）。

## 6. 兜底

- 单文件异常 → 顶层 `try/except`，仍写出合法 JSON。
- 单文件预算 8 s，整体预算 540 s（低于评测 600 s 上限）。
- 解析失败的文件跳过，不影响其它文件。
