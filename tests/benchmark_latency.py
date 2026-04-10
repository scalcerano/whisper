"""Detailed latency breakdown per component."""

import time
import numpy as np
from whisper.audio import load_audio, SAMPLE_RATE
from whisper.streaming import StreamingTranscriber
from whisper.vad import VoiceActivityDetector


def main():
    audio = load_audio("tests/jfk.flac")

    # 1. Pre-warm model
    print("Pre-warming model...")
    t0 = time.perf_counter()
    transcriber = StreamingTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        language="en",
        beam_size=5,
    )
    transcriber._load_model()

    # Warm up GPU with a dummy transcription
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    dummy[1000:5000] = np.sin(np.linspace(0, 100, 4000)).astype(np.float32) * 0.3
    transcriber._model.transcribe(dummy, language="en", beam_size=5, without_timestamps=True)
    warmup_ms = (time.perf_counter() - t0) * 1000
    print(f"Model pre-warmed in {warmup_ms:.0f}ms")

    # 2. Extract speech segments using VAD
    print("\n--- VAD BENCHMARK ---")
    vad = VoiceActivityDetector(threshold=0.3, min_silence_ms=500)

    t0 = time.perf_counter()
    chunk_size = int(SAMPLE_RATE * 0.1)
    segments = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        segs = vad.process(chunk)
        segments.extend(segs)
    segs = vad.flush()
    segments.extend(segs)
    vad_ms = (time.perf_counter() - t0) * 1000
    print(f"VAD: {vad_ms:.0f}ms for {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"  Segment {i}: {len(seg)/SAMPLE_RATE*1000:.0f}ms")

    # 3. Benchmark transcription per segment (post-warmup)
    print("\n--- TRANSCRIPTION BENCHMARK ---")
    for i, seg in enumerate(segments):
        dur_ms = len(seg) / SAMPLE_RATE * 1000

        t0 = time.perf_counter()
        result_segs, info = transcriber._model.transcribe(
            seg,
            language="en",
            beam_size=5,
            without_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text_parts = [s.text.strip() for s in result_segs]
        text = " ".join(text_parts)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"  Seg {i}: {dur_ms:.0f}ms audio -> {latency_ms:.0f}ms latency -> \"{text}\"")

    # 4. Run 5 iterations on segment 1 to get stable latency
    if segments:
        print("\n--- STABLE LATENCY (5 runs on segment 1) ---")
        seg = segments[0]
        latencies = []
        for run in range(5):
            t0 = time.perf_counter()
            result_segs, _ = transcriber._model.transcribe(
                seg, language="en", beam_size=5,
                without_timestamps=True, vad_filter=False,
            )
            _ = [s.text for s in result_segs]
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            print(f"  Run {run+1}: {lat:.0f}ms")

        avg = np.mean(latencies)
        p50 = np.median(latencies)
        p95 = np.percentile(latencies, 95)
        print(f"  Avg: {avg:.0f}ms | P50: {p50:.0f}ms | P95: {p95:.0f}ms")


if __name__ == "__main__":
    main()
