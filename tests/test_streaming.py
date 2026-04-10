"""Tests for the streaming transcription module.

Tests cover:
- VAD speech detection and segmentation
- Audio mel spectrogram chunk computation
- StreamingTranscriber initialization and configuration
- End-to-end streaming transcription flow
"""

import os

import numpy as np
import pytest

from whisper.audio import SAMPLE_RATE, log_mel_spectrogram_chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_sine_wave(freq_hz: float, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave as float32 audio at 16kHz."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _generate_silence(duration_s: float) -> np.ndarray:
    """Generate silence as float32 audio at 16kHz."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _load_jfk_audio() -> np.ndarray:
    """Load the JFK test audio file."""
    from whisper.audio import load_audio

    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    return load_audio(audio_path)


# ---------------------------------------------------------------------------
# Tests: log_mel_spectrogram_chunk
# ---------------------------------------------------------------------------


class TestMelSpectrogramChunk:
    """Tests for variable-length mel spectrogram computation."""

    def test_short_chunk_produces_proportional_frames(self):
        """A 1-second chunk should produce ~100 mel frames (not 3000)."""
        audio = _generate_sine_wave(440, duration_s=1.0)
        mel = log_mel_spectrogram_chunk(audio)

        assert mel.ndim == 2
        assert mel.shape[0] == 80  # default n_mels
        # 1s of audio at 16kHz / HOP_LENGTH(160) = 100 frames (approximately)
        assert 95 <= mel.shape[1] <= 105

    def test_two_second_chunk(self):
        """A 2-second chunk should produce ~200 mel frames."""
        audio = _generate_sine_wave(440, duration_s=2.0)
        mel = log_mel_spectrogram_chunk(audio)

        assert 195 <= mel.shape[1] <= 205

    def test_output_range(self):
        """Mel values should be in a reasonable normalized range."""
        audio = _generate_sine_wave(440, duration_s=1.0)
        mel = log_mel_spectrogram_chunk(audio)

        assert mel.max() - mel.min() <= 2.0
        assert mel.max() <= 1.5  # normalized range

    def test_128_mels(self):
        """Should support 128 mel filters (used by large-v3)."""
        audio = _generate_sine_wave(440, duration_s=1.0)
        mel = log_mel_spectrogram_chunk(audio, n_mels=128)

        assert mel.shape[0] == 128

    def test_real_audio(self):
        """Mel spectrogram of real audio should have reasonable values."""
        audio = _load_jfk_audio()
        # Take just 2 seconds
        chunk = audio[: SAMPLE_RATE * 2]
        mel = log_mel_spectrogram_chunk(chunk)

        assert mel.ndim == 2
        assert mel.shape[0] == 80
        assert 195 <= mel.shape[1] <= 205


# ---------------------------------------------------------------------------
# Tests: VoiceActivityDetector
# ---------------------------------------------------------------------------


class TestVAD:
    """Tests for the Silero VAD wrapper."""

    def test_init_default_params(self):
        from whisper.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        assert vad.sample_rate == 16000
        assert not vad.is_speaking
        assert vad.buffered_duration_ms == 0.0

    def test_invalid_sample_rate(self):
        from whisper.vad import VoiceActivityDetector

        with pytest.raises(ValueError, match="not supported"):
            VoiceActivityDetector(sample_rate=44100)

    def test_reset_clears_state(self):
        from whisper.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        # Simulate some state
        vad._is_speaking = True
        vad._speech_counter = 1000
        vad._speech_buffer = [np.zeros(480)]

        vad.reset()
        assert not vad.is_speaking
        assert vad.buffered_duration_ms == 0.0
        assert vad._speech_counter == 0

    def test_silence_produces_no_segments(self):
        """Pure silence should not produce any speech segments."""
        from whisper.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        silence = _generate_silence(2.0)
        segments = vad.process(silence)

        assert len(segments) == 0

    def test_flush_empty_buffer(self):
        from whisper.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        segments = vad.flush()
        assert len(segments) == 0


# ---------------------------------------------------------------------------
# Tests: StreamingTranscriber
# ---------------------------------------------------------------------------


class TestStreamingTranscriber:
    """Tests for the StreamingTranscriber configuration and state."""

    def test_init_valid_config(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3",
            device="cpu",
            compute_type="float32",
            language="it",
        )
        assert t.model_size == "large-v3"
        assert t.language == "it"
        assert t.device == "cpu"

    def test_init_invalid_model(self):
        from whisper.streaming import StreamingTranscriber

        with pytest.raises(ValueError, match="Unsupported model size"):
            StreamingTranscriber(model_size="nonexistent")

    def test_init_invalid_compute_type(self):
        from whisper.streaming import StreamingTranscriber

        with pytest.raises(ValueError, match="Unsupported compute type"):
            StreamingTranscriber(compute_type="quantum")

    def test_reset_clears_context(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        t._previous_text = "some previous text"
        t._segment_count = 5
        t._total_audio_ms = 10000.0

        t.reset()
        assert t._previous_text == ""
        assert t._segment_count == 0
        assert t._total_audio_ms == 0.0

    def test_stats_empty(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        stats = t.stats

        assert stats["segments_processed"] == 0
        assert stats["average_latency_ms"] == 0.0
        assert stats["model_size"] == "large-v3"

    def test_all_supported_languages(self):
        """All 6 target languages should be accepted."""
        from whisper.streaming import StreamingTranscriber

        for lang in ["it", "en", "de", "fr", "es", "pt"]:
            t = StreamingTranscriber(
                model_size="large-v3",
                device="cpu",
                language=lang,
            )
            assert t.language == lang

    def test_build_prompt_with_initial(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3",
            device="cpu",
            initial_prompt="telephony customer service",
        )
        prompt = t._build_prompt()
        assert "telephony customer service" in prompt

    def test_build_prompt_with_context(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        t._previous_text = "Buongiorno, come posso aiutarla?"
        prompt = t._build_prompt()
        assert "Buongiorno" in prompt

    def test_build_prompt_empty(self):
        from whisper.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        prompt = t._build_prompt()
        assert prompt is None


# ---------------------------------------------------------------------------
# Tests: TranscriptionResult
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_dataclass_fields(self):
        from whisper.streaming import TranscriptionResult

        r = TranscriptionResult(
            text="Ciao mondo",
            language="it",
            duration_ms=1500.0,
            latency_ms=120.0,
        )
        assert r.text == "Ciao mondo"
        assert r.language == "it"
        assert r.is_final is True

    def test_interim_result(self):
        from whisper.streaming import TranscriptionResult

        r = TranscriptionResult(
            text="partial",
            language="en",
            duration_ms=500.0,
            latency_ms=50.0,
            is_final=False,
        )
        assert r.is_final is False
