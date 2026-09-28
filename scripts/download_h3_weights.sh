#!/usr/bin/env bash
# ==============================================================
# YYC³ 漫剧生产系统 · MiniMax-H3 NF4 权重下载器（hf-mirror 镜像 · 断点续传）
# 对齐：docs/legacy/Mac-M4-Max-128GB-MiniMax-H3-NF4完整部署脚本.md 模型清单
# 产物落位：yyc3-minimax-h3/models/h3/（models/ 已在 .gitignore，不入仓库）
# 用法：bash scripts/download_h3_weights.sh [目标目录，默认 models/h3]
# 断点续传：中断后重复执行本脚本即可（huggingface-cli 原生 resume）
# ==============================================================
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-"${SCRIPT_DIR}/../models/h3"}"
mkdir -p "${DEST}/nf4" "${DEST}/processor"

echo "[1/3] 镜像端点：${HF_ENDPOINT}"
echo "[2/3] 磁盘空间检查（需 ~35Gi 可用）"
AVAIL_GB=$(df -g "${DEST}" | tail -1 | awk '{print $4}')
if [ "${AVAIL_GB}" -lt 35 ]; then
  echo "可用空间仅 ${AVAIL_GB}Gi（<35Gi），中止。请清理后重试。"
  exit 1
fi
echo "      可用 ${AVAIL_GB}Gi，通过"

echo "[3/3] 下载 NF4 权重（DiffSynth-Studio/MiniMax-H3-NF4，~30Gi）"
huggingface-cli download "DiffSynth-Studio/MiniMax-H3-NF4" \
  --include "minimax-h3-fl2va-nf4.safetensors" \
            "minimax-h3-ref2va-nf4.safetensors" \
            "minimax-h3-text-encoder-nf4.safetensors" \
            "video_vae_nf4.safetensors" \
            "audio_vae_nf4.safetensors" \
  --local-dir "${DEST}/nf4" \
  --local-dir-use-symlinks False

echo "下载 processor 配置（MiniMax/MiniMax-H3：FL2VA + Ref2VA）"
huggingface-cli download "MiniMax/MiniMax-H3" \
  --include "FL2VA/processor/*" "Ref2VA/processor/*" \
  --local-dir "${DEST}/processor" \
  --local-dir-use-symlinks False

echo "完成。产物清单："
find "${DEST}" -type f -exec ls -lh {} \; | awk '{print $5, $9}'
