#!/bin/bash
#=============================================================================
# EntropyGuard 一键部署 & 实验脚本
# 用法：
#   bash setup_and_run.sh            # 默认：模拟熵全量 + 真熵验证
#   bash setup_and_run.sh fast       # 快速：只跑模拟熵，跳过 embedding
#   bash setup_and_run.sh real       # 真熵：跑真实 LLM 推理取熵（需 API Key）
#   bash setup_and_run.sh cross      # 跨模型：3 个模型对比实验（需 API Key）
#   bash setup_and_run.sh paper      # 论文全量：模拟 + 真熵 + 跨模型全跑
#=============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR]${NC}   $*"; }

MODE="${1:-full}"
HF_MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
SF_API_KEY="${SILICONFLOW_API_KEY:-}"

# =========================================================================
# STEP 1: 环境检查
# =========================================================================
log "========== STEP 1: 环境检查 =========="
echo "项目目录: $PROJECT_DIR"
echo "Python:   $(python3 --version 2>&1 || echo 'NOT FOUND')"
echo "pip:      $(pip3 --version 2>&1 || echo 'NOT FOUND')"
echo "模式:     $MODE"

if python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    log "PyTorch OK"
else
    warn "PyTorch 未安装，将在下一步安装"
fi

# =========================================================================
# STEP 2: 配置 SiliconFlow（如果需要 API 模式）
# =========================================================================
log "========== STEP 2: 配置 SiliconFlow =========="

NEED_API=false
if [[ "$MODE" == "real" || "$MODE" == "cross" || "$MODE" == "paper" ]]; then
    NEED_API=true
fi

if [ -f ".env" ]; then
    log ".env 已存在"
    if grep -q "SILICONFLOW_API_KEY=sk-" .env 2>/dev/null; then
        log "SiliconFlow Key 已配置"
    elif [ "$NEED_API" = true ]; then
        warn ".env 中未检测到 SiliconFlow Key，真熵实验将失败"
    fi
else
    cp .env.example .env
    if [ -n "$SF_API_KEY" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|SILICONFLOW_API_KEY=|SILICONFLOW_API_KEY=$SF_API_KEY|" .env
        else
            sed -i "s|SILICONFLOW_API_KEY=|SILICONFLOW_API_KEY=$SF_API_KEY|" .env
        fi
        log "SiliconFlow Key 已写入 .env"
    elif [ "$NEED_API" = true ]; then
        err "需要 SILICONFLOW_API_KEY。请设置环境变量后重试："
        err "  SILICONFLOW_API_KEY=sk-xxx bash setup_and_run.sh $MODE"
        err "  或手动编辑 .env:  vim .env"
        exit 1
    else
        warn "未设置 SiliconFlow Key（模拟熵模式不需要）"
    fi
fi

# =========================================================================
# STEP 3: 安装依赖
# =========================================================================
log "========== STEP 3: 安装依赖 =========="

PIP_BREAK=""
if python3 -c "import sys; exit(0 if sys.platform=='darwin' else 1)" 2>/dev/null; then
    PIP_BREAK="--break-system-packages"
fi

if [[ "$MODE" == "fast" || "$MODE" == "full" ]]; then
    log "安装依赖（清华镜像）..."
    pip3 install -r requirements.txt -i "$PIP_MIRROR" -q $PIP_BREAK
else
    log "安装全部依赖..."
    pip3 install -r requirements.txt -i "$PIP_MIRROR" $PIP_BREAK
    pip3 install huggingface_hub -i "$PIP_MIRROR" -q $PIP_BREAK
fi

log "依赖安装完成"

# =========================================================================
# STEP 4: 下载完整数据集
# =========================================================================
log "========== STEP 4: 下载 ClawTrojan 数据集 =========="

DATASET_DIR="$PROJECT_DIR/claw_trojan/envs_full"

if [ -d "$DATASET_DIR" ] && [ "$(ls "$DATASET_DIR" 2>/dev/null | wc -l)" -gt 20 ]; then
    log "数据集已存在 ($DATASET_DIR)，跳过下载"
else
    log "从 Hugging Face 下载完整数据集 (362 样本, ~91MB)..."
    export HF_ENDPOINT="$HF_MIRROR"

    HF_REPO=$(python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('zstanjj/ClawTrojan', repo_type='dataset')
print(path)
")

    log "数据集下载到: $HF_REPO"
    ENVS_SRC="$HF_REPO/envs"
    if [ -d "$ENVS_SRC" ]; then
        ln -sfn "$ENVS_SRC" "$DATASET_DIR"
        log "软链接: $DATASET_DIR -> $ENVS_SRC"
    else
        err "数据集目录结构异常"
        ls "$HF_REPO"
        exit 1
    fi
fi

SAMPLE_COUNT=$(ls -d "$DATASET_DIR"/*/ 2>/dev/null | wc -l | tr -d ' ')
log "数据集样本数: $SAMPLE_COUNT"

# =========================================================================
# STEP 5: 运行实验
# =========================================================================
log "========== STEP 5: 运行 EntropyGuard 实验 =========="

OUTPUT_DIR="$PROJECT_DIR/results/entropy_guard"
mkdir -p "$OUTPUT_DIR"

# --- 公用：模拟熵评估 ---
run_simulated() {
    log ">> 模拟熵主实验"
    python3 run_entropy_guard.py evaluate \
        --envs-root "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR"

    log ">> 逐样本检测"
    python3 run_entropy_guard.py detect \
        --envs-root "$DATASET_DIR" \
        --verbose | tee "$OUTPUT_DIR/detection_log.txt"

    log ">> 逃逸攻击变体生成"
    python3 run_entropy_guard.py escape-gen \
        --envs-root "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR/escape_data"
}

# --- 真熵提取 ---
run_real() {
    log ">> 真熵提取（DeepSeek-V4-Flash，可能会花 ¥10-15）"
    python3 run_entropy_guard.py real-entropy \
        --envs-root "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR/real" \
        --model "deepseek-ai/DeepSeek-V4-Flash"

    log ">> 用真熵跑 EntropyGuard 评估"
    python3 run_entropy_guard.py evaluate \
        --envs-root "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR/real"
}

# --- 跨模型对比 ---
run_cross() {
    log ">> 跨模型对比（3 模型 × 20 样本，约 ¥3）"
    python3 run_entropy_guard.py cross-model \
        --envs-root "$DATASET_DIR" \
        --output-dir "$OUTPUT_DIR/cross_model" \
        --max-samples 20
}

# --- 验证模式（先跑 5 个看信号）---
run_validate() {
    log ">> 快速验证（5 样本，约 ¥0.2）"
    python3 run_entropy_guard.py validate \
        --envs-root "$DATASET_DIR"
}

case "$MODE" in
    fast)
        run_simulated
        ;;
    full)
        run_simulated
        if [ -n "${SILICONFLOW_API_KEY:-}" ] || grep -q "SILICONFLOW_API_KEY=sk-" .env 2>/dev/null; then
            run_validate
        fi
        ;;
    real)
        run_simulated
        run_real
        ;;
    cross)
        run_simulated
        run_cross
        ;;
    paper)
        log "===== 论文全量实验 ====="
        log "阶段 1/4: 模拟熵基线"
        run_simulated

        log "阶段 2/4: 真熵验证（5 样本）"
        run_validate

        log "阶段 3/4: 真熵全量"
        run_real

        log "阶段 4/4: 跨模型对比"
        run_cross

        log "===== 全部实验完成 ====="
        ;;
    *)
        err "未知模式: $MODE"
        echo "用法: bash setup_and_run.sh [fast|full|real|cross|paper]"
        exit 1
        ;;
esac

# =========================================================================
# STEP 6: 结果摘要
# =========================================================================
log "========== STEP 6: 结果摘要 =========="

if [ -f "$OUTPUT_DIR/summary_table.txt" ]; then
    echo ""
    cat "$OUTPUT_DIR/summary_table.txt"
    echo ""
fi

if [ -f "$OUTPUT_DIR/entropy_guard_results.json" ]; then
    log "完整结果 JSON: $OUTPUT_DIR/entropy_guard_results.json"
    echo ""
    echo "=== 主要指标 ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/entropy_guard_results.json') as f:
    d = json.load(f)
for name in ['DASGuard-only', 'Entropy-only', 'EntropyGuard']:
    if name in d:
        s = d[name]['step']
        print(f'{name}: F1={s[\"f1\"]:.4f}  Recall={s[\"recall\"]:.4f}  Precision={s[\"precision\"]:.4f}  FPR={s[\"fpr\"]:.4f}')
" 2>/dev/null || true
fi

# Cross-model results
if [ -f "$OUTPUT_DIR/cross_model/cross_model_results.json" ]; then
    echo ""
    echo "=== 跨模型对比 ==="
    python3 -c "
import json
with open('$OUTPUT_DIR/cross_model/cross_model_results.json') as f:
    d = json.load(f)
for k, v in sorted(d.items()):
    s = v['step']
    print(f'{k}: F1={s[\"f1\"]:.4f}  Recall={s[\"recall\"]:.4f}')
" 2>/dev/null || true
fi

log "========== 全部完成 =========="
echo "输出目录: $OUTPUT_DIR"
