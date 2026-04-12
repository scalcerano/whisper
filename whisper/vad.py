"""Voice Activity Detection using Silero VAD.

Provides speech boundary detection for streaming transcription.
Detects when the user starts and stops speaking, producing semantically
coherent audio segments instead of fixed-size chunks.
"""

from typing import List

import numpy as np
import torch


class VoiceActivityDetector:
    """Silero VAD wrapper optimized for telephony streaming.

    Processes audio in small frames (typically 30ms) and tracks speech state
    to produce speech segments bounded by natural pauses.

    Parameters
    ----------
    threshold : float
        Speech probability threshold (0.0-1.0). Higher = stricter detection.
    min_speech_ms : int
        Minimum speech duration in ms to trigger a segment.
    min_silence_ms : int
        Minimum silence duration in ms to end a segment.
    max_speech_ms : int
        Maximum speech duration before forcing a segment boundary.
    sample_rate : int
        Audio sample rate (must be 16000 for Silero VAD).
    """

    SUPPORTED_SAMPLE_RATES = {8000, 16000}
    FRAME_SIZE_MS = 32  # Silero VAD requires >= 512 samples at 16kHz (sr/31.25)

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 300,
        max_speech_ms: int = 5000,
        sample_rate: int = 16000,
    ):
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Sample rate {sample_rate} not supported. "
                f"Use one of {self.SUPPORTED_SAMPLE_RATES}"
            )

        self.threshold = threshold
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.max_speech_samples = int(max_speech_ms * sample_rate / 1000)
        self.sample_rate = sample_rate
        self.frame_size = int(self.FRAME_SIZE_MS * sample_rate / 1000)

        # Pre-speech tolerance: allow brief silence gaps while accumulating
        # enough speech frames to trigger _is_speaking. Without this,
        # natural micro-pauses between words reset the accumulator and
        # speech is never detected.
        self._pre_speech_silence_limit = int(100 * sample_rate / 1000)  # 100ms

        self._model = None
        self._speech_buffer: List[np.ndarray] = []
        self._silence_counter = 0
        self._speech_counter = 0
        self._is_speaking = False
        self._global_offset = 0  # total samples processed

    def _load_model(self):
        """Lazy-load Silero VAD model on first use."""
        if self._model is not None:
            return

        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

    def _get_speech_prob(self, audio_frame: np.ndarray) -> float:
        """Get speech probability for a single frame."""
        tensor = torch.from_numpy(audio_frame).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate)

        return prob.item() if torch.is_tensor(prob) else float(prob)

    def process(self, audio_chunk: np.ndarray) -> List[np.ndarray]:
        """Process an audio chunk and return completed speech segments.

        Parameters
        ----------
        audio_chunk : np.ndarray
            Audio samples as float32, shape (n_samples,).

        Returns
        -------
        list of np.ndarray
            Completed speech segments. Empty list if no segment is complete yet.
        """
        self._load_model()

        completed_segments = []

        # Process frame by frame
        for i in range(0, len(audio_chunk), self.frame_size):
            frame = audio_chunk[i : i + self.frame_size]
            if len(frame) < self.frame_size:
                # Pad last frame if too short
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            prob = self._get_speech_prob(frame)

            # Use the actual slice length (not padded frame) for accurate counting
            actual_len = min(self.frame_size, len(audio_chunk) - i)

            if prob >= self.threshold:
                # Speech detected
                self._silence_counter = 0
                self._speech_counter += actual_len
                self._speech_buffer.append(audio_chunk[i : i + actual_len])

                if not self._is_speaking:
                    if self._speech_counter >= self.min_speech_samples:
                        self._is_speaking = True

                # Force segment boundary on max duration
                if self._speech_counter >= self.max_speech_samples:
                    segment = np.concatenate(self._speech_buffer)
                    completed_segments.append(segment)
                    self._speech_buffer = []
                    self._speech_counter = 0
                    self._is_speaking = False

            else:
                # Silence detected
                self._silence_counter += actual_len

                if self._is_speaking:
                    self._speech_buffer.append(audio_chunk[i : i + actual_len])

                    if self._silence_counter >= self.min_silence_samples:
                        # End of speech segment
                        segment = np.concatenate(self._speech_buffer)
                        completed_segments.append(segment)
                        self._speech_buffer = []
                        self._speech_counter = 0
                        self._silence_counter = 0
                        self._is_speaking = False

                elif self._speech_counter > 0:
                    # Pre-speech phase: tolerate brief silence gaps
                    self._speech_buffer.append(audio_chunk[i : i + actual_len])

                    if self._silence_counter >= self._pre_speech_silence_limit:
                        # Too much silence, discard pre-speech accumulation
                        self._speech_counter = 0
                        self._silence_counter = 0
                        self._speech_buffer = []

        self._global_offset += len(audio_chunk)
        return completed_segments

    def flush(self) -> List[np.ndarray]:
        """Flush any remaining speech in the buffer.

        Call this when the audio stream ends to get the final segment.

        Returns
        -------
        list of np.ndarray
            Remaining speech segment, or empty list.
        """
        segments = []
        if self._speech_buffer and self._is_speaking:
            segment = np.concatenate(self._speech_buffer)
            if len(segment) >= self.min_speech_samples:
                segments.append(segment)

        self._speech_buffer = []
        self._speech_counter = 0
        self._silence_counter = 0
        self._is_speaking = False
        return segments

    def reset(self):
        """Reset all state for a new conversation."""
        self._speech_buffer = []
        self._silence_counter = 0
        self._speech_counter = 0
        self._is_speaking = False
        self._global_offset = 0
        if self._model is not None:
            self._model.reset_states()

    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently being detected."""
        return self._is_speaking

    @property
    def buffered_duration_ms(self) -> float:
        """Duration of audio currently buffered, in milliseconds."""
        if not self._speech_buffer:
            return 0.0
        total_samples = sum(len(b) for b in self._speech_buffer)
        return total_samples / self.sample_rate * 1000
