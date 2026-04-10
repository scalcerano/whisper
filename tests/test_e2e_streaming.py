"""End-to-end streaming test with real audio on GPU."""

import time
import numpy as np
from whisper.audio import load_audio, SAMPLE_RATE
from whisper.streaming import StreamingTranscriber


def main():
    audio = load_audio("tests/jfk.flac")
    print(f"Audio: {len(audio)/SAMPLE_RATE:.1f}s")

    print("Loading large-v3 int8 cuda...")
    transcriber = StreamingTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        language="en",
        beam_size=5,
        vad_threshold=0.3,
        min_silence_ms=500,
    )

    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks

    all_results = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        results = transcriber.feed(chunk)
        for r in results:
            all_results.append(r)
            print(f"  [{r.latency_ms:.0f}ms] ({r.duration_ms:.0f}ms audio) {r.text}")

    for r in transcriber.flush():
        all_results.append(r)
        print(f"  [{r.latency_ms:.0f}ms] ({r.duration_ms:.0f}ms audio) [flush] {r.text}")

    s = transcriber.stats
    seg = s["segments_processed"]
    avg = s["average_latency_ms"]
    ta = s["total_audio_ms"]
    tl = s["total_latency_ms"]
    rtf = tl / ta if ta > 0 else 0

    print()
    print("=== RESULTS ===")
    print(f"Segments: {seg}")
    print(f"Avg latency: {avg:.0f}ms")
    print(f"Total audio: {ta:.0f}ms")
    print(f"Total latency: {tl:.0f}ms")
    print(f"RTF: {rtf:.3f}")

    if seg == 0:
        print("\nWARNING: No segments detected. VAD may need threshold tuning.")


if __name__ == "__main__":
    main()
