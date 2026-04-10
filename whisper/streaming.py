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

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .audio import SAMPLE_RATE, log_mel_spectrogram_chunk
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

        self._ct2_model = None  # ctranslate2.models.Whisper
        self._tokenizer = None  # faster_whisper tokenizer
        self._vad = VoiceActivityDetector(
            threshold=vad_threshold,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            max_speech_ms=max_speech_ms,
            sample_rate=SAMPLE_RATE,
        )

        # Number of mel filters — 128 for large-v3/turbo, 80 for others
        self._n_mels = 128 if model_size in {
            "large-v3", "large", "large-v3-turbo", "turbo",
        } else 80

        # Context tracking for cross-segment coherence
        self._previous_tokens: List[int] = []
        self._segment_count: int = 0
        self._total_audio_ms: float = 0.0
        self._total_latency_ms: float = 0.0

    def _load_model(self):
        """Lazy-load the CTranslate2 Whisper model on first use.

        Uses faster-whisper only for model download and conversion to
        CTranslate2 format. All inference goes through the CTranslate2
        API directly, bypassing faster-whisper's 30s padding.
        """
        if self._ct2_model is not None:
            return

        try:
            import ctranslate2
            from faster_whisper.utils import download_model
            from faster_whisper.tokenizer import Tokenizer
        except ImportError:
            raise ImportError(
                "faster-whisper and ctranslate2 are required for streaming. "
                "Install with: pip install faster-whisper"
            )

        # Download/convert model via faster-whisper, get the CTranslate2 path
        model_path = download_model(self.model_size)

        # Load CTranslate2 model directly — this is the C++ engine
        self._ct2_model = ctranslate2.models.Whisper(
            model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

        # Build tokenizer for prompt encoding and text decoding
        from tokenizers import Tokenizer as HFTokenizer

        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        hf_tokenizer = HFTokenizer.from_file(tokenizer_path)
        self._tokenizer = Tokenizer(
            hf_tokenizer,
            self._ct2_model.is_multilingual,
            task="transcribe",
            language=self.language,
        )

    def _build_prompt_tokens(self) -> List[int]:
        """Build the decoder prompt token sequence.

        Returns the SOT (start-of-transcript) sequence with language,
        task, and optional context from previous segments.
        """
        # SOT sequence: <|startoftranscript|> [<|lang|>] <|transcribe|> <|notimestamps|>
        tokens = list(self._tokenizer.sot_sequence)
        tokens.append(self._tokenizer.no_timestamps)

        # Prepend previous context for cross-segment coherence
        if self._previous_tokens:
            prev = self._previous_tokens[-200:]  # limit context length
            tokens = [self._tokenizer.sot_prev] + prev + tokens

        # Prepend initial prompt if set
        if self.initial_prompt:
            prompt_tokens = self._tokenizer.encode(" " + self.initial_prompt.strip())
            tokens = [self._tokenizer.sot_prev] + prompt_tokens + tokens

        return tokens

    def _transcribe_segment(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a speech segment via CTranslate2 directly.

        Computes a mel spectrogram proportional to the actual audio
        duration (no 30s padding) and passes it straight to the
        CTranslate2 C++ encoder and decoder. This is the key
        optimization: a 2s segment produces ~200 mel frames instead
        of the 3000 frames that faster-whisper would pad to.
        """
        self._load_model()

        t_start = time.perf_counter()
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        # Compute mel spectrogram — proportional to actual duration, no padding
        mel = log_mel_spectrogram_chunk(audio, n_mels=self._n_mels)

        # Shape for CTranslate2: (batch=1, n_mels, n_frames)
        features = np.expand_dims(mel.numpy(), axis=0).astype(np.float32)

        # Encode audio — C++ kernel, processes only the real frames
        import ctranslate2

        features_sv = ctranslate2.StorageView.from_array(features)

        # Detect language if not configured
        detected_language = self.language
        if detected_language is None:
            lang_results = self._ct2_model.detect_language(features_sv)
            detected_language = lang_results[0][0][0]  # top language
            # Update tokenizer with detected language
            self._tokenizer.language_code = detected_language

        # Build prompt and generate
        prompt_tokens = self._build_prompt_tokens()

        # Limit max_length proportionally to audio duration to prevent
        # the decoder from generating far more text than the audio contains.
        # Whisper produces ~25 tokens/second of speech.
        duration_s = len(audio) / SAMPLE_RATE
        max_tokens = max(int(duration_s * 30), 10)  # 30 tok/s with margin

        results = self._ct2_model.generate(
            features_sv,
            [prompt_tokens],
            beam_size=self.beam_size,
            max_length=max_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            suppress_blank=True,
            suppress_tokens=[-1],
            return_no_speech_prob=True,
        )

        # Decode tokens to text
        result = results[0]
        output_tokens = result.sequences_ids[0]

        # Filter out special tokens and decode
        text_tokens = [
            t for t in output_tokens
            if t < self._tokenizer.eot
        ]
        text = self._tokenizer.decode(text_tokens).strip()

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Update context for next segment
        if text_tokens:
            self._previous_tokens = text_tokens
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
        self._previous_tokens = []
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
