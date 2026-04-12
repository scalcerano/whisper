# Whisper — Streaming Fork

> Fork of [openai/whisper](https://github.com/openai/whisper) with **real-time streaming transcription** for telephony systems.

[[Original Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition, speech translation, and language identification.

**This fork adds a streaming STT module** targeting sub-200ms latency per segment, designed for real-time telephony in Italian, English, German, French, Spanish, and Portuguese.


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

    pip install -U openai-whisper

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/openai/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```


## Available models and languages

There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model.
The relative speeds below are measured by transcribing English speech on a A100, and the real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3` that offers faster transcription speed with a minimal degradation in accuracy.

Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## Command-line usage

The following command will transcribe speech in audio files, using the `turbo` model:

```bash
whisper audio.flac audio.mp3 audio.wav --model turbo
```

The default setting (which selects the `turbo` model) works well for transcribing English. However, **the `turbo` model is not trained for translation tasks**. If you need to **translate non-English speech into English**, use one of the **multilingual models** (`tiny`, `base`, `small`, `medium`, `large`) instead of `turbo`. 

For example, to transcribe an audio file containing non-English speech, you can specify the language:

```bash
whisper japanese.wav --language Japanese
```

To **translate** speech into English, use:

```bash
whisper japanese.wav --model medium --language Japanese --task translate
```

> **Note:** The `turbo` model will return the original language even if `--task translate` is specified. Use `medium` or `large` for the best translation results.

Run the following to view all available options:

```bash
whisper --help
```

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.


## Python usage

Transcription can also be performed within Python: 

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model.

```python
import whisper

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

## Streaming transcription

This fork adds a streaming STT module optimized for real-time telephony. It combines **Silero VAD** for speech boundary detection with **CTranslate2** for low-latency GPU inference.

### Requirements

- **Python** 3.9+
- **NVIDIA GPU** with CUDA (tested on L4, A10, T4)
- **ffmpeg** installed and in PATH
- ~6 GB VRAM for large-v3-turbo INT8

### Installation

```bash
# 1. Install this fork
pip install git+https://github.com/scalcerano/whisper.git

# 2. Install streaming dependencies
pip install faster-whisper torchaudio

# 3. Install PyTorch with CUDA (adjust cu124 to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify the installation:

```bash
python -c "from whisper.streaming import StreamingTranscriber; print('OK')"
```

### Quick start

```python
from whisper import StreamingTranscriber

# Initialize (model is lazy-loaded on first feed())
transcriber = StreamingTranscriber(
    model_size="large-v3-turbo",
    device="cuda",
    compute_type="int8",
    language="it",
)

# Feed audio as it arrives (16kHz float32 PCM, e.g. from a telephony channel)
for pcm_chunk in audio_stream:
    results = transcriber.feed(pcm_chunk)
    for r in results:
        print(f"[{r.latency_ms:.0f}ms] {r.text}")

# Flush remaining audio at end of call
for r in transcriber.flush():
    print(f"[final] {r.text}")

# Reset for next call (model stays loaded in VRAM)
transcriber.reset()
```

### Transcribing a file in streaming mode

```python
from whisper.audio import load_audio, SAMPLE_RATE
from whisper.streaming import StreamingTranscriber

transcriber = StreamingTranscriber(
    model_size="large-v3-turbo",
    device="cuda",
    compute_type="int8",
    language="it",
)

# Load audio file (any format ffmpeg supports)
audio = load_audio("call_recording.wav")

# Feed in 100ms chunks — simulates real-time telephony input
chunk_size = int(SAMPLE_RATE * 0.1)  # 1600 samples = 100ms
for i in range(0, len(audio), chunk_size):
    chunk = audio[i : i + chunk_size]
    for result in transcriber.feed(chunk):
        print(f"[{result.latency_ms:.0f}ms] {result.text}")

# Get any remaining text
for result in transcriber.flush():
    print(f"[flush] {result.text}")

# Print statistics
stats = transcriber.stats
print(f"Segments: {stats['segments_processed']}")
print(f"Avg latency: {stats['average_latency_ms']:.0f}ms")
```

### Architecture

```
Audio 16kHz float32 PCM
  |
  v
VoiceActivityDetector (Silero VAD)
  - Detects speech start/end using neural VAD
  - Produces segments bounded by natural pauses (1-8 seconds)
  - Filters silence — only speech reaches the transcriber
  |
  v
Pad to 30 seconds + log-mel spectrogram
  - Raw audio padded with silence to 480,000 samples (30s)
  - STFT + mel filterbank = 3000 mel frames (matches Whisper training)
  - Critical: shorter mel degrades encoder output quality
  |
  v
CTranslate2 Whisper encoder (INT8)
  - 32 transformer layers, 128 mel filters
  - Produces 1500 encoder positions
  |
  v
CTranslate2 Whisper decoder (turbo: 4 layers, beam search)
  - Cross-attention over 1500 encoder positions
  - Domain prompt carried across segments for coherence
  - no_speech_prob filter prevents hallucinations on VAD false positives
  |
  v
TranscriptionResult(text, language, duration_ms, latency_ms)
```

### Latency breakdown (NVIDIA L4, INT8)

| Component | Time |
|-----------|------|
| Silero VAD | ~5ms |
| Mel spectrogram (30s pad) | ~1ms |
| CTranslate2 encoder | ~138ms |
| CTranslate2 decoder (turbo) | ~50ms |
| **Total per segment** | **~190ms** |

### Benchmark results

Tested on NVIDIA L4 GPU, INT8 quantization, beam_size=5.

**1289 Italian audiobook segments (Dumas, Pirandello):**

| Model | WER | Good (<30%) | Bad (>60%) | Latency |
|-------|-----|-------------|------------|---------|
| **large-v3-turbo** (default) | **10.5%** | 94.6% | 0.3% | **294ms** |
| large-v3 | 12.0% | 93.0% | 1.0% | 457ms |

**40 min real conversational Italian (live audience, applause, emotion):**
- 491 speech segments detected
- 188ms average latency per segment
- 0 hallucinated segments (no_speech filter active)

### Configuration reference

```python
StreamingTranscriber(
    # Model
    model_size="large-v3-turbo", # default; "large-v3" for max accuracy (~450ms)
    device="cuda",               # "cuda" or "cpu"
    compute_type="int8",         # int8, float16, int8_float16, float32

    # Language
    language="it",               # ISO code; None for auto-detection per segment

    # Decoder
    beam_size=5,                 # beam search width (1=greedy, 5=default)

    # VAD — controls how speech segments are detected
    vad_threshold=0.5,           # speech probability threshold (0.0-1.0)
    min_speech_ms=400,           # minimum speech duration to trigger transcription
    min_silence_ms=600,          # silence duration to end a speech segment
    max_speech_ms=8000,          # maximum segment length before forced split

    # Prompting — helps with domain-specific vocabulary
    initial_prompt="immobiliare, mutuo, rogito",  # domain terms
    telephony_hints=True,        # adds email/address/phone spelling vocabulary

    # Interim results — partial transcriptions during active speech
    interim_interval_ms=1000,    # emit interim every 1s of speech (None=disabled)
)
```

### Interim results

By default, the system waits for end-of-speech (silence) before emitting text.
With `interim_interval_ms`, you get partial results while the user is still talking:

```python
transcriber = StreamingTranscriber(
    model_size="large-v3-turbo",
    device="cuda",
    compute_type="int8",
    language="it",
    interim_interval_ms=1000,  # emit partial results every 1 second of speech
)

for pcm_chunk in audio_stream:
    for r in transcriber.feed(pcm_chunk):
        if r.is_final:
            print(f"[FINAL] {r.text}")
        else:
            print(f"[interim conf={r.confidence:.2f}] {r.text}")
```

Interim results have `is_final=False` and don't affect cross-segment context.
Each interim replaces the previous one. The final result is the definitive version.

### Confidence scores

Every `TranscriptionResult` includes a confidence score for downstream decision-making:

```python
for r in transcriber.feed(chunk):
    if r.confidence < 0.3:
        ask_caller_to_repeat()
    elif r.confidence > 0.8:
        process_with_high_confidence(r.text)
```

### Audio format handling

`feed()` accepts any common audio format — no pre-processing needed:

```python
# 8kHz int16 from PSTN — auto-resampled and converted
transcriber.feed(pstn_audio_int16, sample_rate=8000)

# 48kHz float32 from VoIP — auto-resampled
transcriber.feed(voip_audio_float32, sample_rate=48000)

# Stereo — auto-mixed to mono
transcriber.feed(stereo_audio)
```

### Concurrent calls (model pool)

Multiple `StreamingTranscriber` instances with the same model config automatically
share a single model in VRAM. Each instance keeps its own VAD, language, and context:

```python
# These two share one model in VRAM (~6GB total, not 12GB)
call_1 = StreamingTranscriber(language="it")
call_2 = StreamingTranscriber(language="en")

# Independent state — each call has its own VAD and context
call_1.feed(audio_from_caller_1)
call_2.feed(audio_from_caller_2)
```

### TranscriptionResult fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | str | Transcribed text |
| `language` | str | Language code (e.g. "it") |
| `duration_ms` | float | Duration of the speech segment |
| `latency_ms` | float | Processing time (audio received to text produced) |
| `confidence` | float | Transcription confidence 0.0–1.0 (from decoder logprob) |
| `no_speech_prob` | float | Probability of no speech 0.0–1.0 |
| `is_final` | bool | True for completed segments, False for interim partials |

### Supported languages

All Whisper-supported languages work. Primary targets for telephony:
Italian, English, German, French, Spanish, Portuguese.

### Troubleshooting

**"No module named 'ctranslate2'"** — Install faster-whisper: `pip install faster-whisper`

**"No module named 'torchaudio'"** — Install torchaudio: `pip install torchaudio`

**ffmpeg not found** — Install ffmpeg: `apt install ffmpeg` (Ubuntu) or `brew install ffmpeg` (macOS)

**CUDA out of memory** — large-v3-turbo needs ~6 GB VRAM with INT8. Use `compute_type="int8"` or a smaller model.

**High latency on first segment** — Expected. The first `feed()` call lazy-loads the model (~5s) and runs a CUDA warmup inference. Subsequent segments are ~190ms.

**Empty transcriptions** — Segments with `no_speech_prob > 0.6` are filtered to prevent hallucinations. If too many segments are filtered, lower `vad_threshold` to reduce false positives.

### Running the benchmark

```bash
# On a GPU machine with Italian test audio in data/italian_segments/
python tests/test_italian_streaming.py --samples 20 --model large-v3-turbo
python tests/test_italian_streaming.py --all  # run all samples
```

## More examples

Please use the [Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) category in Discussions for sharing more example usages of Whisper and third-party extensions such as web demos, integrations with other tools, ports for different platforms, etc.


## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

All dependencies (CTranslate2, faster-whisper, Silero VAD, PyTorch, tiktoken) are MIT or BSD licensed.
