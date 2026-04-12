"""Test streaming transcription with Italian audio."""

import numpy as np
from whisperng.audio import load_audio, SAMPLE_RATE
from whisperng.streaming import StreamingTranscriber


def main():
    audio = load_audio("tests/lella.wav")
    dur = len(audio) / SAMPLE_RATE
    print(f"Audio: {dur:.1f}s, {len(audio)} samples")

    t = StreamingTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        language="it",
        beam_size=5,
        vad_threshold=0.3,
        min_silence_ms=500,
    )

    chunk_size = int(SAMPLE_RATE * 0.1)
    all_r = []
    for i in range(0, len(audio), chunk_size):
        for r in t.feed(audio[i : i + chunk_size]):
            all_r.append(r)
            print(f"  [{r.latency_ms:.0f}ms] ({r.duration_ms:.0f}ms) {r.text}")

    for r in t.flush():
        all_r.append(r)
        print(f"  [{r.latency_ms:.0f}ms] ({r.duration_ms:.0f}ms) [flush] {r.text}")

    s = t.stats
    tl = s["total_latency_ms"]
    ta = s["total_audio_ms"]
    rtf = tl / ta if ta > 0 else 0
    print()
    print(f"Segments: {s['segments_processed']}")
    print(f"Avg latency: {s['average_latency_ms']:.0f}ms")
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
