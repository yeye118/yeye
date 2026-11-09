#!/bin/bash
# 选手方案入口（评测服务经 Dockerfile 的 ENTRYPOINT 调用本脚本，并追加参数）。
set -e
echo "=== 反思式决策智能体启动 ==="
echo "输入数据目录 DATA_ROOT = $DATA_ROOT"
echo "运行参数: $*"

cd /app/program/solution
exec python3 src/main.py "$@"
