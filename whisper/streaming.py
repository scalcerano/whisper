"""Streaming speech-to-text transcription using Whisper with CTranslate2.

This module provides a ``StreamingTranscriber`` that processes audio in
real-time using VAD-guided chunking and the CTranslate2 inference backend
for low-latency transcription (~190ms per segment on L4 GPU with INT8).

Typical usage for telephony::

    transcriber = StreamingTranscriber(
        model_size="large-v3-turbo",
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

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("whisper.streaming")

from .audio import N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
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
    confidence : float
        Transcription confidence (0.0–1.0). Derived from the decoder's
        average log-probability. Values above 0.6 are generally reliable;
        below 0.3 suggest the system should ask the caller to repeat.
    no_speech_prob : float
        Probability that the segment contains no speech (0.0–1.0).
        High values (>0.6) indicate the VAD passed noise or silence
        that the decoder recognized as non-speech.
    is_final : bool
        Whether this is a final result (True) or interim (False).
    """

    text: str
    language: str
    duration_ms: float
    latency_ms: float
    confidence: float = 0.0
    no_speech_prob: float = 0.0
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
        Whisper model size. Default "large-v3-turbo" — same encoder as
        large-v3 with a 4-layer decoder for ~190ms latency on L4 GPU.
        Use "large-v3" (32-layer decoder, ~450ms) when max accuracy is needed.
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
        When ``telephony_hints`` is True, a telephony-specific prompt is
        prepended automatically.
    telephony_hints : bool
        When True, prepend a prompt with telephony vocabulary (email
        spelling, addresses, phone numbers) to improve recognition of
        dictated contact information.
    """

    # Short domain vocabulary prompt — anchors the decoder without bloating
    # the token budget. Whisper doesn't need verbose instructions, just
    # key terms it might encounter in a telephony context.
    _TELEPHONY_PROMPT = (
        "Telefonata. Email: chiocciola @, punto. "
        "Telefono, indirizzo, codice fiscale, partita IVA."
    )

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "int8",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_threshold: float = 0.5,
        min_speech_ms: int = 400,
        min_silence_ms: int = 600,
        max_speech_ms: int = 8000,
        initial_prompt: Optional[str] = None,
        telephony_hints: bool = False,
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

        # Build effective prompt: telephony hints + user prompt
        prompt_parts = []
        if telephony_hints:
            prompt_parts.append(self._TELEPHONY_PROMPT)
        if initial_prompt:
            prompt_parts.append(initial_prompt)
        self.initial_prompt = " ".join(prompt_parts) if prompt_parts else None

        self._ct2_model = None   # ctranslate2.models.Whisper
        self._tokenizer = None   # faster_whisper tokenizer
        self._StorageView = None # ctranslate2.StorageView (cached at load)
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

        # Cache StorageView to avoid per-segment import lookup
        self._StorageView = ctranslate2.StorageView

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

        # Warm up CUDA kernels with a FULL 3000-frame mel — must match
        # production shape so JIT compilation covers the real code path.
        # A 300-frame warmup is useless: different shape = different kernels.
        dummy_mel = np.zeros((1, self._n_mels, N_FRAMES), dtype=np.float32)
        dummy_features = self._StorageView.from_array(dummy_mel)
        prompt = list(self._tokenizer.sot_sequence)
        prompt.append(self._tokenizer.no_timestamps)
        self._ct2_model.generate(
            dummy_features, [prompt],
            beam_size=self.beam_size,
            max_length=1,
        )

    def _build_prompt_tokens(self) -> List[int]:
        """Build the decoder prompt token sequence.

        Returns the SOT (start-of-transcript) sequence with language,
        task, and optional context from previous segments.

        Token budget: Whisper decoder has 448 max tokens total.
        We must leave room for the actual transcription output.
        Budget: prompt ≤ 80 tokens (~18% of decoder capacity).

        When initial_prompt is set, it is carried across ALL segments
        (not just the first) to maintain domain conditioning — critical
        for telephony where email/phone dictation can happen at any point.
        """
        MAX_PROMPT_TOKENS = 80  # leave 368 tokens for generated text

        # SOT sequence: <|startoftranscript|> [<|lang|>] <|transcribe|> <|notimestamps|>
        sot = list(self._tokenizer.sot_sequence)
        sot.append(self._tokenizer.no_timestamps)

        # Budget for prefix tokens (everything before SOT sequence)
        budget = MAX_PROMPT_TOKENS - len(sot) - 1  # -1 for sot_prev token

        prefix = []
        if self.initial_prompt:
            # Domain prompt — capped at 15 tokens to leave room for context.
            # Carried on EVERY segment so telephony vocabulary (chiocciola,
            # punto, codice fiscale) is always available to the decoder.
            ip_tokens = self._tokenizer.encode(" " + self.initial_prompt.strip())
            ip_tokens = ip_tokens[:15]

            if self._previous_tokens:
                # Both: domain prompt + previous transcript for coherence
                remaining = budget - len(ip_tokens)
                prev = self._previous_tokens[-remaining:] if remaining > 0 else []
                prefix = [self._tokenizer.sot_prev] + ip_tokens + prev
            else:
                prefix = [self._tokenizer.sot_prev] + ip_tokens
        elif self._previous_tokens:
            prev = self._previous_tokens[-budget:]
            prefix = [self._tokenizer.sot_prev] + prev

        return prefix + sot

    def _transcribe_segment(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe a speech segment via CTranslate2 directly.

        Pads audio to 30 seconds and computes a standard 3000-frame mel
        spectrogram, matching Whisper's training format. The encoder was
        trained exclusively on 30s windows; shorter input degrades
        comprehension because the positional embeddings and cross-attention
        were calibrated for 1500 encoder positions.
        """
        self._load_model()

        t_start = time.perf_counter()
        duration_ms = len(audio) / SAMPLE_RATE * 1000

        # Pad raw audio to 30 seconds THEN compute mel. This is critical:
        # 1. Whisper encoder expects exactly 3000 mel frames (→ 1500 positions)
        # 2. Padding raw audio with silence produces correct low-energy mel
        #    values, unlike zero-padding the mel directly
        # 3. On L4 GPU INT8 the encoder processes 3000 frames in ~138ms
        audio_padded = pad_or_trim(audio, N_SAMPLES)
        mel = log_mel_spectrogram(audio_padded, n_mels=self._n_mels)

        # Shape for CTranslate2: (batch=1, n_mels, n_frames)
        features = np.expand_dims(mel.numpy(), axis=0).astype(np.float32)
        features_sv = self._StorageView.from_array(features)

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
        # Guard: CTranslate2 requires max_length > 0 after internal prompt
        # accounting — ensure a safe minimum that exceeds any prompt size.
        duration_s = len(audio) / SAMPLE_RATE
        max_tokens = max(int(duration_s * 40), len(prompt_tokens) + 10)

        t_mel = time.perf_counter()
        logger.info(f"[STT-DBG] mel shape={features.shape}, prompt_len={len(prompt_tokens)}, max_tokens={max_tokens}, duration={duration_s:.1f}s")

        results = self._ct2_model.generate(
            features_sv,
            [prompt_tokens],
            beam_size=self.beam_size,
            max_length=max_tokens,
            repetition_penalty=1.1,
            suppress_blank=True,
            suppress_tokens=[-1],
            return_no_speech_prob=True,
            return_scores=True,
        )

        t_gen = time.perf_counter()

        # Extract decoder metrics for confidence scoring
        result = results[0]
        seg_no_speech = result.no_speech_prob if hasattr(result, 'no_speech_prob') else 0.0
        n_tokens = len(result.sequences_ids[0]) if result.sequences_ids and result.sequences_ids[0] else 0

        # Confidence = exp(avg_logprob). CTranslate2 scores[0] is the
        # cumulative log-probability for the top beam. Normalizing by
        # token count gives avg_logprob; exp() maps to 0.0–1.0 range.
        import math
        if hasattr(result, 'scores') and result.scores and n_tokens > 0:
            avg_logprob = result.scores[0] / n_tokens
            seg_confidence = math.exp(max(avg_logprob, -10.0))  # clamp to avoid underflow
        else:
            seg_confidence = 0.0

        logger.info(
            f"[STT-DBG] generate: {(t_gen-t_mel)*1000:.0f}ms, tokens={n_tokens}, "
            f"no_speech={seg_no_speech:.3f}, confidence={seg_confidence:.3f}"
        )

        # Helper to build empty results (no-speech filter or empty decode)
        def _empty_result() -> TranscriptionResult:
            lat = (time.perf_counter() - t_start) * 1000
            self._segment_count += 1
            self._total_audio_ms += duration_ms
            self._total_latency_ms += lat
            return TranscriptionResult(
                text="",
                language=detected_language,
                duration_ms=duration_ms,
                latency_ms=lat,
                confidence=seg_confidence,
                no_speech_prob=seg_no_speech,
                is_final=True,
            )

        # Guard: if decoder says "no speech" with high confidence, the VAD
        # made a false positive. Return empty rather than hallucinate.
        if seg_no_speech > 0.6:
            logger.info(f"[STT-DBG] no_speech_prob={seg_no_speech:.3f} > 0.6 — skipping segment")
            return _empty_result()

        if not result.sequences_ids or not result.sequences_ids[0]:
            return _empty_result()

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
            confidence=seg_confidence,
            no_speech_prob=seg_no_speech,
            is_final=True,
        )

    @staticmethod
    def _normalize_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize audio to float32 mono at 16kHz.

        Handles common telephony formats without requiring the caller
        to pre-process:
        - int16 → float32 (divide by 32768)
        - stereo → mono (average channels)
        - any sample rate → 16kHz (torchaudio resample)

        Parameters
        ----------
        audio : np.ndarray
            Raw audio samples (float32/int16, mono/stereo).
        sample_rate : int
            Input sample rate.

        Returns
        -------
        np.ndarray
            Float32 mono audio at 16kHz.
        """
        # int16 → float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1 if audio.shape[1] <= 2 else 0)
        elif audio.ndim > 2:
            audio = audio.reshape(-1)

        # resample to 16kHz if needed
        if sample_rate != SAMPLE_RATE:
            import torch
            try:
                import torchaudio.functional as F
                audio_t = torch.from_numpy(audio)
                audio_t = F.resample(audio_t, sample_rate, SAMPLE_RATE)
                audio = audio_t.numpy()
            except ImportError:
                raise ImportError(
                    f"torchaudio is required for resampling from {sample_rate}Hz. "
                    "Install with: pip install torchaudio"
                )

        return audio

    def feed(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> List[TranscriptionResult]:
        """Feed an audio chunk and get transcription results.

        Call this method with each incoming audio frame from the telephony
        channel. Internally, audio is buffered and processed through VAD.
        When a complete speech segment is detected (speech followed by
        sufficient silence), it is transcribed and returned.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Audio samples. Accepts float32 or int16, mono or stereo.
            Automatically resampled to 16kHz if ``sample_rate`` differs.
        sample_rate : int
            Sample rate of the input audio (default: 16000). Common
            telephony rates: 8000 (PSTN), 16000 (wideband), 48000 (VoIP).

        Returns
        -------
        list of TranscriptionResult
            Transcription results for any completed speech segments.
            Empty list if no segment is complete yet.
        """
        audio_chunk = self._normalize_audio(audio_chunk, sample_rate)
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
