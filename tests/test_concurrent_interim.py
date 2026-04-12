"""Test concurrent sessions with interim results on GPU."""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisperng.audio import SAMPLE_RATE, load_audio
from whisperng.streaming import StreamingTranscriber


def run_session(session_id, audio, chunk_size, results_out, errors_out):
    try:
        t = StreamingTranscriber(
            model_size="large-v3-turbo",
            device="cuda",
            compute_type="int8",
            language="it",
            interim_interval_ms=1000,
        )
        session_results = []
        for i in range(0, len(audio), chunk_size):
            for r in t.feed(audio[i : i + chunk_size]):
                session_results.append(r)
        for r in t.flush():
            session_results.append(r)

        stats = t.stats
        results_out[session_id] = {
            "results": session_results,
            "avg_latency": stats["average_latency_ms"],
            "segments": stats["segments_processed"],
        }
    except Exception as e:
        import traceback
        errors_out.append(traceback.format_exc())


def main():
    data_dir = "/opt/whisper/data/italian_segments"
    files = [
        os.path.join(data_dir, "dumas_002_dumas_montecristo_rr_i_0000.wav"),
        os.path.join(data_dir, "dumas_002_dumas_montecristo_rr_i_0064.wav"),
        os.path.join(data_dir, "dumas_003_dumas_montecristo_rr_ii_0005.wav"),
    ]
    audios = [load_audio(f) for f in files]
    chunk_size = int(SAMPLE_RATE * 0.1)

    results_out = {}
    errors_out = []

    print("=== 3 CONCURRENT SESSIONS + INTERIM (1000ms) ===")
    print()

    t0 = time.perf_counter()
    threads = []
    for i in range(3):
        th = threading.Thread(target=run_session, args=(i, audios[i], chunk_size, results_out, errors_out))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    wall_ms = (time.perf_counter() - t0) * 1000

    if errors_out:
        print("ERRORS:")
        for e in errors_out:
            print(e)
        print()

    for sid in sorted(results_out):
        d = results_out[sid]
        rlist = d["results"]
        finals = [r for r in rlist if r.is_final]
        interims = [r for r in rlist if not r.is_final]
        avg_lat = d["avg_latency"]
        print(f"Session {sid}: {len(finals)} final, {len(interims)} interim, avg_lat={avg_lat:.0f}ms")
        for r in rlist:
            tag = "FINAL" if r.is_final else "inter"
            print(f"  [{tag} c={r.confidence:.2f} {r.latency_ms:5.0f}ms] {r.text[:80]}")
        print()

    print(f"Wall time (3 concurrent): {wall_ms:.0f}ms")
    print(f"Model cache entries: {len(StreamingTranscriber._model_cache)}")

    # Verify model sharing
    assert len(StreamingTranscriber._model_cache) == 1, "Expected 1 cached model"
    assert len(results_out) == 3, "Expected 3 sessions"
    assert len(errors_out) == 0, f"Unexpected errors: {errors_out}"

    total_finals = sum(len([r for r in d["results"] if r.is_final]) for d in results_out.values())
    total_interims = sum(len([r for r in d["results"] if not r.is_final]) for d in results_out.values())
    print(f"\nTotal: {total_finals} final, {total_interims} interim across 3 sessions")
    print("PASS" if total_finals > 0 and total_interims > 0 else "FAIL — no interim results!")


if __name__ == "__main__":
    main()
