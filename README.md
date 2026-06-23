<h1 align="center">EntropyGuard</h1>

<p align="center">
Entropy-Guided Dual-Channel Defense Against Persistent Prompt Injection in LLM Agents
</p>

<div align="center">
<a href="https://huggingface.co/datasets/zstanjj/ClawTrojan" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Dataset-27b3b4.svg" alt="HuggingFace Dataset"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
<a><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"></a>
</div>

## Overview

LLM agents operating in local workspaces can read/write files, invoke tools, and reuse state across sessions, making them vulnerable to **persistent prompt injection**—malicious instructions planted in workspace files that gradually hijack agent behavior across multiple interaction steps.

**EntropyGuard** is a dual-channel defense framework that bridges model-internal entropy signals with workspace-level text pattern detection through an adaptive threshold mechanism. When an LLM processes injected control instructions, its output entropy exhibits distinctive low-stability patterns. EntropyGuard leverages this signal to dynamically modulate the sensitivity of text-pattern scanning, catching attacks that single-layer defenses miss.

### Key Features

- **Dual-Channel Detection**: Entropy monitoring (InFliGuard sliding-window) + DASGuard text scanning
- **Adaptive Threshold**: Entropy anomaly signals dynamically lower text scanner thresholds
- **Contamination Tracking**: Cross-step pollution propagation with ancestral risk scoring
- **Escape Attack Robustness**: Maintains F1=0.985 under both entropy-escape and text-escape attacks
- **Zero False Positives**: Perfect precision (1.00) on full ClawTrojan benchmark (1,671 steps)

## Results

| Method | Precision | Recall | F1 | FPR |
|--------|-----------|--------|----|-----|
| DASGuard-only | 1.00 | 0.71 | 0.83 | 0.00 |
| Entropy-only | 1.00 | 1.00 | 1.00 | 0.00 |
| **EntropyGuard** | **1.00** | **0.97** | **0.99** | **0.00** |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/shihuizhang-dazhi/claw.git
cd claw

# 2. Install
pip install -r requirements.txt

# 3. Download dataset & run experiments
bash setup_and_run.sh fast

# 4. With real entropy (SiliconFlow API key required)
cp .env.example .env
# Edit .env: SILICONFLOW_API_KEY=sk-your-key
bash setup_and_run.sh paper
```

## Usage Modes

```bash
bash setup_and_run.sh fast     # Simulated entropy, 2 min, free
bash setup_and_run.sh full      # Simulated + validation (5 real samples)
bash setup_and_run.sh real      # Full real entropy extraction (~¥15)
bash setup_and_run.sh cross     # Cross-model comparison (3 models)
bash setup_and_run.sh paper     # Everything: sim + real + cross-model
```

Or run individual commands:

```bash
python run_entropy_guard.py evaluate --envs-root claw_trojan/envs_full
python run_entropy_guard.py detect --envs-root claw_trojan/envs_full -v
python run_entropy_guard.py validate --envs-root claw_trojan/envs_full
python run_entropy_guard.py real-entropy --envs-root claw_trojan/envs_full --dry-run
python run_entropy_guard.py cross-model --envs-root claw_trojan/envs_full
```

## Dataset

Uses the [ClawTrojan benchmark](https://huggingface.co/datasets/zstanjj/ClawTrojan) (362 trajectories, 1,671 steps, 7 attack types). The `setup_and_run.sh` script downloads it automatically.

## Architecture

```
Agent generates response
    ├── Entropy Channel: per-token entropy → anomaly score A
    ├── Text Channel: DASGuard scan → risk findings {r_k}
    └── Fusion Engine: τ' = τ_base − α·A
        → reclassify findings with adjusted thresholds
        → EntropyGuardDecision (block/quarantine/preserve)
```

## Project Structure

```
├── entropy_guard/           # EntropyGuard core
│   ├── entropy_channel.py   # InFliGuard sliding-window entropy monitor
│   ├── fusion.py            # Adaptive threshold fusion engine
│   ├── evaluate.py          # Full experiment suite
│   ├── escape_attacks.py    # Cross-layer escape attack generation
│   └── real_entropy.py      # Real entropy extraction from API logprobs
├── agent_eval/              # DASGuard scanner & evaluation
│   ├── dasguard/            # Text-pattern scanning channel
│   └── dti/                 # Drift monitoring (embedding backends)
├── claw_trojan/             # Benchmark loader
│   ├── loader.py            # TrojanStepSample data loader
│   └── envs/                # Test subset (15 samples)
├── run_entropy_guard.py     # CLI entry point
├── setup_and_run.sh         # One-click setup & experiment script
└── paper/                   # LaTeX source & compiled PDF
```

## Requirements

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
- Optional: SiliconFlow API key for real entropy experiments
- No GPU required for simulated entropy mode

## Citation

```bibtex
@misc{entropyguard2026,
  title        = {EntropyGuard: Entropy-Guided Dual-Channel Defense Against Persistent Prompt Injection in LLM Agents},
  author       = {Anonymous},
  year         = {2026},
  note         = {Under Review}
}
```

## Acknowledgements

Built on the [ClawTrojan](https://github.com/RUC-NLPIR/ClawTrojan) benchmark and [DualSentinel](https://arxiv.org/abs/2603.01574)'s InFliGuard algorithm.
