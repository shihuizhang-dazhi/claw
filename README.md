<h1 align="center">EntropyGuard</h1>

<p align="center"><strong>Entropy-Guided Dual-Channel Defense Against Persistent Prompt Injection in LLM Agents</strong></p>

<p align="center">
  <a href="paper/entropy_guard.pdf"><img src="https://img.shields.io/badge/Paper-PDF-blue" alt="Paper PDF"></a>
  <a href="https://huggingface.co/datasets/zstanjj/ClawTrojan" target="_blank"><img src="https://img.shields.io/badge/Dataset-ClawTrojan-27b3b4.svg" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

## Overview

LLM agents with tool-use and file-system access are vulnerable to **persistent prompt injection** — hidden instructions planted in workspace files that gradually hijack agent behavior. EntropyGuard is a **dual-channel defense** that fuses two complementary signals:

- **Entropy Channel**: Monitors per-token generation entropy using a sliding-window stability detector. When the model follows injected instructions, its output becomes anomalously low-entropy.
- **Text Channel**: DASGuard scans workspace text for known injection patterns (regex, embedding similarity, fragment-join).
- **Adaptive Fusion**: An entropy anomaly score A dynamically tightens text-scanning thresholds: τ' = τ_base − α·A. This catches borderline injections that each channel alone would miss.

## Key Results

| Method | Precision | Recall | F1 | FPR |
|--------|-----------|--------|----|-----|
| DASGuard-only | 1.00 | 0.71 | 0.83 | 0.00 |
| Entropy-only | 1.00 | 1.00 | 1.00 | 0.00 |
| **EntropyGuard** | **1.00** | **0.98** | **0.99** | **0.00** |

Evaluated on the full ClawTrojan benchmark (362 trajectories, 1,671 steps, 7 attack types). EntropyGuard achieves 0.988 F1 with zero false positives. Entropy hypothesis validated via repeated sampling on a real LLM (GPT-5.4-Mini), confirming malicious outputs are 33% more consistent (lower entropy) than benign outputs.

## Quick Start

```bash
# Clone
git clone https://github.com/shihuizhang-dazhi/claw.git
cd claw

# Install
pip install -r requirements.txt

# Download dataset & run (simulated entropy, free, 2 min)
bash setup_and_run.sh fast

# Full evaluation (simulated entropy + 5-sample real validation)
bash setup_and_run.sh full
```

## Usage

```bash
# Evaluate on full benchmark
python run_entropy_guard.py evaluate --envs-root claw_trojan/envs_full

# Per-sample detection with verbose output
python run_entropy_guard.py detect --envs-root claw_trojan/envs_full -v

# Validate entropy hypothesis (repeated sampling)
python run_entropy_guard.py validate --envs-root claw_trojan/envs_full

# Generate paper figures
python scripts/generate_figures.py

# Compile paper
cd paper && tectonic entropy_guard.tex
```

## Architecture

```
LLM Agent generates response
  ├── Entropy Channel: per-token Shannon entropy → anomaly score A
  ├── Text Channel: DASGuard scan → risk findings {r_k}
  └── Fusion Engine: τ' = τ_base − α·A
      → reclassify findings with adjusted thresholds
      → EntropyGuard decision (block / quarantine / preserve)
```

## Directory Structure

```
├── entropy_guard/           # Core implementation
│   ├── entropy_channel.py   # InFliGuard sliding-window entropy monitor
│   ├── fusion.py            # Adaptive threshold fusion engine
│   ├── evaluate.py          # Full experiment suite
│   ├── escape_attacks.py    # Cross-layer escape attack generation
│   ├── real_entropy.py      # Real entropy extraction from API logprobs
│   └── entropy_validation.py # Hypothesis validation via repeated sampling
├── agent_eval/dasguard/     # DASGuard text-pattern scanner
├── agent_eval/llm_client.py # LLM API clients (OpenAI, SiliconFlow, etc.)
├── claw_trojan/             # ClawTrojan benchmark loader
│   ├── loader.py            # TrojanStepSample data loader
│   └── envs/                # Test subset (15 samples)
├── scripts/
│   └── generate_figures.py  # Paper figure generation from experiment data
├── paper/                   # LaTeX source (IEEEtran)
│   └── entropy_guard.tex    # 10-page paper
├── run_entropy_guard.py     # CLI entry point
├── setup_and_run.sh         # One-click experiment script
└── requirements.txt
```

## Requirements

- Python 3.10+
- ClawTrojan dataset (auto-downloaded by `setup_and_run.sh`)
- Optional: OpenAI API key for real logprobs-based entropy extraction

## Citation

```bibtex
@misc{entropyguard2026,
  title  = {EntropyGuard: Entropy-Guided Dual-Channel Defense Against
            Persistent Prompt Injection in LLM Agents},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgments

- [ClawTrojan](https://github.com/RUC-NLPIR/ClawTrojan) benchmark
- [DualSentinel](https://arxiv.org/abs/2603.01574) InFliGuard algorithm
- [DASGuard](https://github.com/RUC-NLPIR/ClawTrojan) text-pattern scanner
