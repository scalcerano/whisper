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

from whisperng.audio import SAMPLE_RATE, log_mel_spectrogram_chunk


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
    from whisperng.audio import load_audio

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
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        assert vad.sample_rate == 16000
        assert not vad.is_speaking
        assert vad.buffered_duration_ms == 0.0

    def test_invalid_sample_rate(self):
        from whisperng.vad import VoiceActivityDetector

        with pytest.raises(ValueError, match="not supported"):
            VoiceActivityDetector(sample_rate=44100)

    def test_reset_clears_state(self):
        from whisperng.vad import VoiceActivityDetector

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
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        silence = _generate_silence(2.0)
        segments = vad.process(silence)

        assert len(segments) == 0

    def test_flush_empty_buffer(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        segments = vad.flush()
        assert len(segments) == 0


# ---------------------------------------------------------------------------
# Tests: StreamingTranscriber
# ---------------------------------------------------------------------------


class TestStreamingTranscriber:
    """Tests for the StreamingTranscriber configuration and state."""

    def test_init_valid_config(self):
        from whisperng.streaming import StreamingTranscriber

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
        from whisperng.streaming import StreamingTranscriber

        with pytest.raises(ValueError, match="Unsupported model size"):
            StreamingTranscriber(model_size="nonexistent")

    def test_init_invalid_compute_type(self):
        from whisperng.streaming import StreamingTranscriber

        with pytest.raises(ValueError, match="Unsupported compute type"):
            StreamingTranscriber(compute_type="quantum")

    def test_reset_clears_context(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        t._previous_tokens = [1, 2, 3]
        t._segment_count = 5
        t._total_audio_ms = 10000.0

        t.reset()
        assert t._previous_tokens == []
        assert t._segment_count == 0
        assert t._total_audio_ms == 0.0

    def test_stats_empty(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        stats = t.stats

        assert stats["segments_processed"] == 0
        assert stats["average_latency_ms"] == 0.0
        assert stats["model_size"] == "large-v3"

    def test_all_supported_languages(self):
        """All 6 target languages should be accepted."""
        from whisperng.streaming import StreamingTranscriber

        for lang in ["it", "en", "de", "fr", "es", "pt"]:
            t = StreamingTranscriber(
                model_size="large-v3",
                device="cpu",
                language=lang,
            )
            assert t.language == lang

    def test_initial_prompt_stored(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3",
            device="cpu",
            initial_prompt="telephony customer service",
        )
        assert t.initial_prompt == "telephony customer service"

    def test_telephony_hints_prompt(self):
        """telephony_hints=True should prepend telephony vocabulary."""
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3",
            device="cpu",
            telephony_hints=True,
        )
        assert "chiocciola" in t.initial_prompt
        assert "@" in t.initial_prompt

    def test_telephony_hints_combined_with_user_prompt(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3",
            device="cpu",
            telephony_hints=True,
            initial_prompt="immobiliare",
        )
        assert "chiocciola" in t.initial_prompt
        assert "immobiliare" in t.initial_prompt

    def test_no_telephony_hints_default(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        assert t.initial_prompt is None

    def test_previous_tokens_empty_at_init(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        assert t._previous_tokens == []

    def test_n_mels_large_v3(self):
        """Large-v3 uses 128 mel filters."""
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3", device="cpu")
        assert t._n_mels == 128

    def test_n_mels_small(self):
        """Smaller models use 80 mel filters."""
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="small", device="cpu")
        assert t._n_mels == 80


# ---------------------------------------------------------------------------
# Tests: TranscriptionResult
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_dataclass_fields(self):
        from whisperng.streaming import TranscriptionResult

        r = TranscriptionResult(
            text="Ciao mondo",
            language="it",
            duration_ms=1500.0,
            latency_ms=120.0,
        )
        assert r.text == "Ciao mondo"
        assert r.language == "it"
        assert r.is_final is True
        assert r.confidence == 0.0  # default
        assert r.no_speech_prob == 0.0  # default

    def test_confidence_and_no_speech(self):
        from whisperng.streaming import TranscriptionResult

        r = TranscriptionResult(
            text="test",
            language="it",
            duration_ms=1000.0,
            latency_ms=100.0,
            confidence=0.85,
            no_speech_prob=0.02,
        )
        assert r.confidence == 0.85
        assert r.no_speech_prob == 0.02

    def test_interim_result(self):
        from whisperng.streaming import TranscriptionResult

        r = TranscriptionResult(
            text="partial",
            language="en",
            duration_ms=500.0,
            latency_ms=50.0,
            is_final=False,
        )
        assert r.is_final is False


# ---------------------------------------------------------------------------
# Tests: Audio normalization
# ---------------------------------------------------------------------------


class TestAudioNormalization:
    """Tests for automatic audio format normalization."""

    def test_int16_to_float32(self):
        from whisperng.streaming import StreamingTranscriber

        audio_int16 = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        result = StreamingTranscriber._normalize_audio(audio_int16, 16000)
        assert result.dtype == np.float32
        assert abs(result[1] - 0.5) < 0.001
        assert abs(result[3] - 1.0) < 0.001

    def test_stereo_to_mono(self):
        from whisperng.streaming import StreamingTranscriber

        stereo = np.array([[0.5, -0.5], [0.3, -0.3]], dtype=np.float32)
        result = StreamingTranscriber._normalize_audio(stereo, 16000)
        assert result.ndim == 1
        assert abs(result[0]) < 0.001  # (0.5 + -0.5) / 2

    def test_passthrough_float32_16khz(self):
        from whisperng.streaming import StreamingTranscriber

        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = StreamingTranscriber._normalize_audio(audio, 16000)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, audio)

    def test_resample_8khz(self):
        from whisperng.streaming import StreamingTranscriber

        # 8kHz sine wave, 0.1 seconds = 800 samples
        t = np.linspace(0, 0.1, 800, dtype=np.float32)
        audio_8k = np.sin(2 * np.pi * 440 * t)
        result = StreamingTranscriber._normalize_audio(audio_8k, 8000)
        # Should have ~1600 samples (16kHz * 0.1s)
        assert 1550 <= len(result) <= 1650
        assert result.dtype == np.float32

    def test_resample_48khz(self):
        from whisperng.streaming import StreamingTranscriber

        # 48kHz, 0.1 seconds = 4800 samples
        t = np.linspace(0, 0.1, 4800, dtype=np.float32)
        audio_48k = np.sin(2 * np.pi * 440 * t)
        result = StreamingTranscriber._normalize_audio(audio_48k, 48000)
        # Should have ~1600 samples
        assert 1550 <= len(result) <= 1650


# ---------------------------------------------------------------------------
# Tests: VAD peek_buffer
# ---------------------------------------------------------------------------


class TestVADPeekBuffer:
    """Tests for VAD buffer peeking (used by interim results)."""

    def test_peek_empty_buffer(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        assert vad.peek_buffer() is None

    def test_peek_not_speaking(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        vad._speech_buffer = [np.zeros(480)]
        vad._is_speaking = False
        assert vad.peek_buffer() is None  # not confirmed speech

    def test_peek_while_speaking(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        chunk1 = np.ones(512, dtype=np.float32) * 0.1
        chunk2 = np.ones(512, dtype=np.float32) * 0.2
        vad._speech_buffer = [chunk1, chunk2]
        vad._is_speaking = True
        result = vad.peek_buffer()
        assert result is not None
        assert len(result) == 1024
        # Buffer should NOT be consumed
        assert len(vad._speech_buffer) == 2

    def test_speech_duration_ms(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector(sample_rate=16000)
        vad._speech_counter = 16000  # 1 second
        assert vad.speech_duration_ms == 1000.0

    def test_speech_duration_zero(self):
        from whisperng.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        assert vad.speech_duration_ms == 0.0


# ---------------------------------------------------------------------------
# Tests: Interim results configuration
# ---------------------------------------------------------------------------


class TestInterimResults:
    """Tests for interim result configuration and state."""

    def test_interim_disabled_by_default(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(model_size="large-v3-turbo", device="cpu")
        assert t._interim_interval_ms is None

    def test_interim_enabled(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3-turbo",
            device="cpu",
            interim_interval_ms=1000,
        )
        assert t._interim_interval_ms == 1000

    def test_reset_clears_interim_state(self):
        from whisperng.streaming import StreamingTranscriber

        t = StreamingTranscriber(
            model_size="large-v3-turbo",
            device="cpu",
            interim_interval_ms=500,
        )
        t._last_interim_speech_ms = 2000.0
        t.reset()
        assert t._last_interim_speech_ms == 0.0


# ---------------------------------------------------------------------------
# Tests: Model pool (shared cache)
# ---------------------------------------------------------------------------


class TestModelPool:
    """Tests for class-level model cache sharing."""

    def test_cache_key_format(self):
        """Two instances with same config should produce same cache key."""
        from whisperng.streaming import StreamingTranscriber

        t1 = StreamingTranscriber(model_size="large-v3-turbo", device="cpu", compute_type="float32")
        t2 = StreamingTranscriber(model_size="large-v3-turbo", device="cpu", compute_type="float32")
        key1 = f"{t1.model_size}:{t1.device}:{t1.compute_type}"
        key2 = f"{t2.model_size}:{t2.device}:{t2.compute_type}"
        assert key1 == key2

    def test_different_config_different_key(self):
        from whisperng.streaming import StreamingTranscriber

        t1 = StreamingTranscriber(model_size="large-v3-turbo", device="cpu")
        t2 = StreamingTranscriber(model_size="small", device="cpu")
        key1 = f"{t1.model_size}:{t1.device}:{t1.compute_type}"
        key2 = f"{t2.model_size}:{t2.device}:{t2.compute_type}"
        assert key1 != key2

    def test_different_languages_same_model(self):
        """Different languages should share the model but have separate state."""
        from whisperng.streaming import StreamingTranscriber

        t_it = StreamingTranscriber(model_size="large-v3-turbo", device="cpu", language="it")
        t_en = StreamingTranscriber(model_size="large-v3-turbo", device="cpu", language="en")
        assert t_it.language != t_en.language
        # Both would use same cache key (model is language-agnostic)
        key_it = f"{t_it.model_size}:{t_it.device}:{t_it.compute_type}"
        key_en = f"{t_en.model_size}:{t_en.device}:{t_en.compute_type}"
        assert key_it == key_en
        # But they have separate VAD and context
        assert t_it._vad is not t_en._vad
        assert t_it._previous_tokens is not t_en._previous_tokens
