"""Test streaming STT accuracy on Italian audio with known transcriptions.

Processes audio through the full streaming pipeline (VAD → mel 30s → CTranslate2)
in 100ms chunks, simulating real-time telephony input. Compares output against
reference transcriptions and reports WER (Word Error Rate).

Usage:
    python tests/test_italian_streaming.py [--samples N] [--all]
"""

import json
import os
import re
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper.audio import SAMPLE_RATE, load_audio
from whisper.streaming import StreamingTranscriber


# ---------------------------------------------------------------------------
# WER computation (no external dependency)
# ---------------------------------------------------------------------------

def _edit_distance(ref_words, hyp_words):
    """Levenshtein distance between two word lists."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: edit_distance(ref, hyp) / len(ref)."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _normalize(text: str) -> str:
    """Normalize text for WER comparison: lowercase, strip punctuation."""
    text = text.lower().strip()
    # Remove common artifacts from reference transcriptions
    for artifact in ["grazie per la visione", "grazie per la visione!"]:
        text = text.replace(artifact, "")
    # Strip punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test streaming STT on Italian audio")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--all", action="store_true", help="Test all samples")
    parser.add_argument("--data-dir", default="/opt/whisper/data/italian_segments", help="Path to audio files")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load metadata
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith(".json")]
    if not json_files:
        print(f"ERROR: No JSON metadata in {args.data_dir}")
        sys.exit(1)

    json_path = os.path.join(args.data_dir, json_files[0])
    with open(json_path) as f:
        metadata = json.load(f)

    # Select samples
    if args.all:
        samples = metadata
    else:
        # Pick evenly spaced samples for representative coverage
        step = max(1, len(metadata) // args.samples)
        samples = metadata[::step][: args.samples]

    print(f"Testing {len(samples)} samples from {len(metadata)} total")
    print(f"Model: {args.model}, Device: {args.device}")
    print(f"Data dir: {args.data_dir}")
    print("=" * 80)

    # Initialize transcriber
    transcriber = StreamingTranscriber(
        model_size=args.model,
        device=args.device,
        compute_type="int8",
        language="it",
        beam_size=5,
        vad_threshold=0.3,
        min_silence_ms=500,
    )

    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks, like telephony
    total_wer = 0.0
    total_audio_s = 0.0
    total_latency_ms = 0.0
    results = []

    for i, sample in enumerate(samples):
        # Resolve audio path
        basename = os.path.basename(sample["path"])
        audio_path = os.path.join(args.data_dir, basename)
        if not os.path.exists(audio_path):
            print(f"  SKIP: {basename} not found")
            continue

        reference = sample["text"]
        audio = load_audio(audio_path)
        duration_s = len(audio) / SAMPLE_RATE

        # Feed in 100ms chunks — full streaming pipeline
        transcriber.reset()
        all_texts = []
        for j in range(0, len(audio), chunk_size):
            chunk = audio[j: j + chunk_size]
            for r in transcriber.feed(chunk):
                all_texts.append(r.text)

        for r in transcriber.flush():
            all_texts.append(r.text)

        hypothesis = " ".join(all_texts)
        wer = compute_wer(reference, hypothesis)

        total_wer += wer
        total_audio_s += duration_s

        stats = transcriber.stats
        seg_latency = stats["average_latency_ms"]
        total_latency_ms += stats["total_latency_ms"]

        results.append({
            "file": basename,
            "wer": wer,
            "ref": _normalize(reference),
            "hyp": _normalize(hypothesis),
            "duration_s": duration_s,
            "segments": stats["segments_processed"],
            "avg_latency_ms": seg_latency,
        })

        # Status indicator
        status = "OK" if wer < 0.3 else "WARN" if wer < 0.6 else "BAD"
        print(f"[{i+1:3d}/{len(samples)}] WER={wer:.1%} {status}  lat={seg_latency:.0f}ms  seg={stats['segments_processed']}  {basename}")
        if wer > 0.3:
            print(f"         REF: {_normalize(reference)[:120]}")
            print(f"         HYP: {_normalize(hypothesis)[:120]}")

    # Summary
    n = len(results)
    if n == 0:
        print("\nNo samples processed!")
        sys.exit(1)

    avg_wer = total_wer / n
    good = sum(1 for r in results if r["wer"] < 0.3)
    warn = sum(1 for r in results if 0.3 <= r["wer"] < 0.6)
    bad = sum(1 for r in results if r["wer"] >= 0.6)

    print()
    print("=" * 80)
    print(f"RESULTS: {n} samples, {total_audio_s:.0f}s total audio")
    print(f"Average WER:      {avg_wer:.1%}")
    print(f"Good (<30% WER):  {good}/{n}")
    print(f"Warn (30-60%):    {warn}/{n}")
    print(f"Bad  (>60%):      {bad}/{n}")
    print(f"Avg latency/seg:  {total_latency_ms / max(sum(r['segments'] for r in results), 1):.0f}ms")
    print(f"Total latency:    {total_latency_ms:.0f}ms")
    print("=" * 80)

    # Exit code: fail if average WER > 40%
    if avg_wer > 0.4:
        print("\nFAIL: Average WER too high")
        sys.exit(1)
    else:
        print("\nPASS")


if __name__ == "__main__":
    main()
