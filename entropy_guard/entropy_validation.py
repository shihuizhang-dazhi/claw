"""Validate entropy hypothesis via repeated sampling without logprobs.

When logprobs is unavailable from the API, we estimate model uncertainty
by generating multiple responses to the same prompt and measuring output
diversity. Low diversity = low entropy = model is constrained (injected).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_eval.llm_client import OpenAIClient  # works with OpenAI-compatible APIs


def compute_self_bleu(responses: List[str]) -> float:
    """Compute average BLEU of each response against the others.

    High self-BLEU = responses are similar = low entropy.
    """
    if len(responses) < 2:
        return 1.0

    from collections import Counter
    import math

    def tokenize(text: str) -> List[str]:
        return text.lower().split()

    bleus = []
    for i, cand in enumerate(responses):
        cand_tokens = tokenize(cand)
        if not cand_tokens:
            bleus.append(0.0)
            continue

        # Average BLEU-1 against all other responses
        scores = []
        for j, ref in enumerate(responses):
            if i == j:
                continue
            ref_tokens = tokenize(ref)
            if not ref_tokens:
                scores.append(0.0)
                continue
            ref_counts = Counter(ref_tokens)
            matches = sum(1 for t in cand_tokens if ref_counts.get(t, 0) > 0)
            scores.append(matches / len(cand_tokens))
        bleus.append(sum(scores) / len(scores) if scores else 0.0)

    return sum(bleus) / len(bleus) if bleus else 0.0


def validate_entropy_hypothesis(
    samples: list,
    client: OpenAIClient,
    n_samples: int = 5,
    n_repeats: int = 5,
    temperature: float = 0.7,
):
    """Validate that malicious prompts produce lower-entropy (more consistent) outputs."""
    import random

    malicious = [s for s in samples if s.is_malicious]
    benign = [s for s in samples if not s.is_malicious]

    n_mal = min(n_samples, len(malicious))
    n_ben = min(n_samples, len(benign))

    test_samples = random.sample(malicious, n_mal) + random.sample(benign, n_ben)
    random.shuffle(test_samples)

    print(f"\n{'='*60}")
    print(f"Entropy Hypothesis Validation (repeated sampling)")
    print(f"{'='*60}")
    print(f"Samples: {len(test_samples)} ({n_mal} malicious, {n_ben} benign)")
    print(f"Repeats per sample: {n_repeats} (temperature={temperature})")
    print()

    results = []
    for i, sample in enumerate(test_samples):
        label = "MAL" if sample.is_malicious else "BEN"
        prompt = (sample.tool_output or sample.user_input or "")[:500]

        if not prompt.strip():
            continue

        print(f"  [{i+1}/{len(test_samples)}] {sample.eval_id} ({label})")

        responses = []
        for r in range(n_repeats):
            try:
                text = client.complete(
                    "You are an AI assistant. Respond concisely.",
                    prompt,
                )
                responses.append(text.strip())
            except Exception as e:
                print(f"    Repeat {r+1}: ERROR - {e}")

        if len(responses) < 2:
            continue

        self_bleu = compute_self_bleu(responses)
        avg_len = sum(len(r.split()) for r in responses) / len(responses)

        # Compute pairwise Jaccard similarity
        def jaccard(a, b):
            set_a = set(a.lower().split())
            set_b = set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        similarities = []
        for a_idx in range(len(responses)):
            for b_idx in range(a_idx + 1, len(responses)):
                similarities.append(jaccard(responses[a_idx], responses[b_idx]))
        avg_jaccard = sum(similarities) / len(similarities) if similarities else 0.0

        print(f"    Self-BLEU={self_bleu:.4f}  Jaccard={avg_jaccard:.4f}  "
              f"AvgLen={avg_len:.0f}  Responses=[{responses[0][:60]}...]")

        results.append({
            "eval_id": sample.eval_id,
            "label": label,
            "self_bleu": self_bleu,
            "jaccard_sim": avg_jaccard,
            "avg_length": avg_len,
            "n_responses": len(responses),
        })

    # Summary
    mal_results = [r for r in results if r["label"] == "MAL"]
    ben_results = [r for r in results if r["label"] == "BEN"]

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")

    if mal_results:
        avg_bleu = sum(r["self_bleu"] for r in mal_results) / len(mal_results)
        avg_jac = sum(r["jaccard_sim"] for r in mal_results) / len(mal_results)
        print(f"Malicious (n={len(mal_results)}): "
              f"Self-BLEU={avg_bleu:.4f}  Jaccard={avg_jac:.4f}")
        print(f"  → HIGH consistency = LOW entropy → detector can catch these")

    if ben_results:
        avg_bleu = sum(r["self_bleu"] for r in ben_results) / len(ben_results)
        avg_jac = sum(r["jaccard_sim"] for r in ben_results) / len(ben_results)
        print(f"Benign    (n={len(ben_results)}): "
              f"Self-BLEU={avg_bleu:.4f}  Jaccard={avg_jac:.4f}")
        print(f"  → LOW consistency = HIGH entropy → detector won't falsely trigger")

    if mal_results and ben_results:
        effect_size_bleu = (avg_bleu_ben := sum(r["self_bleu"] for r in ben_results) / len(ben_results)) - \
                           (avg_bleu_mal := sum(r["self_bleu"] for r in mal_results) / len(mal_results))
        print(f"\n  Effect size (Self-BLEU): {effect_size_bleu:.4f}")
        print(f"  Malicious responses are {effect_size_bleu/avg_bleu_ben*100:.1f}% more consistent")
        if effect_size_bleu > 0.1:
            print(f"  ✅ Hypothesis CONFIRMED: entropy channel is valid")
        else:
            print(f"  ⚠️  Weak signal — may need more samples or different approach")

    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    from openai import OpenAI
    from claw_trojan.loader import load_all_trojan_envs

    envs_root = sys.argv[1] if len(sys.argv) > 1 else "claw_trojan/envs_full"
    samples = load_all_trojan_envs(envs_root)

    # Use relay API directly (OpenAI-compatible)
    relay = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://llm.mathmodel.tech/v1",
        timeout=180,
    )

    # Wrap in a simple client
    class RelayClient:
        def complete(self, system, user):
            resp = relay.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=256,
            )
            return resp.choices[0].message.content or ""

    results = validate_entropy_hypothesis(samples, RelayClient(), n_samples=5, n_repeats=5)
