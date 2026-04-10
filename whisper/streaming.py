"""Streaming speech-to-text transcription using Whisper with CTranslate2.

This module provides a ``StreamingTranscriber`` that processes audio in
real-time using VAD-guided chunking and the CTranslate2 inference backend
for low-latency transcription (~150ms per segment on GPU with INT8).

Typical usage for telephony::

    transcriber = StreamingTranscriber(
        model_size="large-v3",
        device="cuda",
        compute_type="int8",
        language="it",
    )

    # Feed audio as it arrives from the telephone channel
    for pcm_chunk in audio_stream:
        results = transcriber.feed(pcm_chunk)
        for text in results:
            send_to_pipeline(text)

    # Flush remaining audio at end of call
    final = transcriber.flush()
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .audio import SAMPLE_RATE
from .vad import VoiceActivityDetector


@dataclass
class TranscriptionResult:
    """A single transcription result from a speech segment.

    Attributes
    ----------
    text : str
        Transcribed text.
    language : str
        Detected or configured language code.
    duration_ms : float
        Duration of the speech segment in milliseconds.
    latency_ms : float
        Processing latency in milliseconds (audio received to text produced).
    is_final : bool
        Whether this is a final result (True) or interim (False).
    """

    text: str
    language: str
    duration_ms: float
    latency_ms: float
    is_final: bool = True


# Supported model sizes for CTranslate2 conversion
_CT2_MODEL_SIZES = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo",
}

# Supported compute types for CTranslate2
_CT2_COMPUTE_TYPES = {
    "default", "auto",
    "int8", "int8_float16", "int8_float32", "int8_bfloat16",
    "int16",
    "float16", "bfloat16", "float32",
}


class StreamingTranscriber:
    """Real-time speech-to-text transcriber using Whisper + CTranslate2 + VAD.

    Receives audio in small chunks (e.g. 20-100ms frames from a telephony
    channel), detects speech boundaries using Silero VAD, and transcribes
    completed speech segments using the CTranslate2-optimized Whisper model.

    Parameters
    ----------
    model_size : str
        Whisper model size (e.g. "large-v3", "medium", "small").
    device : str
        Inference device ("cuda" or "cpu").
    compute_type : str
        CTranslate2 compute type (e.g. "int8", "float16", "int8_float16").
    language : str, optional
        Language code (e.g. "it", "en", "de"). If None, auto-detects per segment.
    beam_size : int
        Beam search width. Higher = better quality, more latency.
    vad_threshold : float
        VAD speech probability threshold.
    min_speech_ms : int
        Minimum speech duration to trigger transcription.
    min_silence_ms : int
        Minimum silence duration to end a speech segment.
    max_speech_ms : int
        Maximum speech duration before forcing segment boundary.
    initial_prompt : str, optional
        Initial prompt to condition the decoder for domain-specific vocabulary.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        max_speech_ms: int = 5000,
        initial_prompt: Optional[str] = None,
    ):
        if model_size not in _CT2_MODEL_SIZES:
            raise ValueError(
                f"Unsupported model size '{model_size}'. "
                f"Choose from: {sorted(_CT2_MODEL_SIZES)}"
            )
        if compute_type not in _CT2_COMPUTE_TYPES:
            raise ValueError(
                f"Unsupported compute type '{compute_type}'. "
                f"Choose from: {sorted(_CT2_COMPUTE_TYPES)}"
            )

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt

        self._model = None
        self._vad = VoiceActivityDetector(
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            max_speech_ms=max_speech_ms,
            sample_rate=SAMPLE_RATE,
        )

        # Context tracking for cross-segment coherence
        self._previous_text: str = ""
        self._segment_count: int = 0
        self._total_audio_ms: float = 0.0
        self._total_latency_ms: float = 0.0

    def _load_model(self):
        """Lazy-load the CTranslate2 Whisper model on first use."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for streaming transcription. "
                "Install it with: pip install faster-whisper"
            )

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def _build_prompt(self) -> Optional[str]:
        """Build decoder prompt from initial prompt and previous context."""
        parts = []

        if self.initial_prompt:
            parts.append(self.initial_prompt)

        if self._previous_text:
            # Carry last segment's text as context for coherence
            # Truncate to avoid exceeding decoder context window
            prev = self._previous_text[-200:]
            parts.append(prev)

        return " ".join(parts) if parts else None

    def _transcribe_segment(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a single speech segment using CTranslate2.

        Parameters
        ----------
        audio : np.ndarray
            Speech audio segment as float32 at 16kHz.

        Returns
        -------
        TranscriptionResult
            Transcription with timing metadata.
        """
        self._load_model()

        t_start = time.perf_counter()
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        prompt = self._build_prompt()

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            initial_prompt=prompt,
            vad_filter=False,  # We handle VAD ourselves
            without_timestamps=True,  # Skip timestamps for speed
            condition_on_previous_text=False,  # We manage context ourselves
        )

        # Consume the generator to get all text
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts)
        detected_language = info.language if self.language is None else self.language

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Update context
        if text:
            self._previous_text = text
        self._segment_count += 1
        self._total_audio_ms += duration_ms
        self._total_latency_ms += latency_ms

        return TranscriptionResult(
            text=text,
            language=detected_language,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            is_final=True,
        )

    def feed(self, audio_chunk: np.ndarray) -> List[TranscriptionResult]:
        """Feed an audio chunk and get transcription results.

        Call this method with each incoming audio frame from the telephony
        channel. Internally, audio is buffered and processed through VAD.
        When a complete speech segment is detected (speech followed by
        sufficient silence), it is transcribed and returned.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Audio samples as float32 at 16kHz. Typical frame sizes for
            telephony: 20ms (320 samples) or 30ms (480 samples).

        Returns
        -------
        list of TranscriptionResult
            Transcription results for any completed speech segments.
            Empty list if no segment is complete yet.
        """
        speech_segments = self._vad.process(audio_chunk)
        results = []

        for segment_audio in speech_segments:
            result = self._transcribe_segment(segment_audio)
            if result.text:  # Skip empty transcriptions
                results.append(result)

        return results

    def flush(self) -> List[TranscriptionResult]:
        """Flush remaining audio and return final transcription.

        Call this at the end of a conversation to transcribe any
        audio still in the VAD buffer.

        Returns
        -------
        list of TranscriptionResult
            Final transcription results.
        """
        remaining = self._vad.flush()
        results = []

        for segment_audio in remaining:
            result = self._transcribe_segment(segment_audio)
            if result.text:
                results.append(result)

        return results

    def reset(self):
        """Reset all state for a new conversation.

        Clears the VAD buffer, context, and statistics. The model stays
        loaded — only conversation state is reset.
        """
        self._vad.reset()
        self._previous_text = ""
        self._segment_count = 0
        self._total_audio_ms = 0.0
        self._total_latency_ms = 0.0

    @property
    def stats(self) -> dict:
        """Get transcription statistics for the current conversation.

        Returns
        -------
        dict
            Statistics including segment count, total audio duration,
            total and average latency.
        """
        avg_latency = (
            self._total_latency_ms / self._segment_count
            if self._segment_count > 0
            else 0.0
        )
        return {
            "segments_processed": self._segment_count,
            "total_audio_ms": self._total_audio_ms,
            "total_latency_ms": self._total_latency_ms,
            "average_latency_ms": avg_latency,
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "device": self.device,
            "language": self.language,
        }
