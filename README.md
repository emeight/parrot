# Parrot

![Parrot](assets/spix's_macaw.jpg)
> In 2018 Spix's Macaw was declared extinct in the wild, but reintroduction efforts are ongoing in Brazil.

On-device speech-to-text and text-to-speech, fully local, no cloud APIs.

## Models

- **STT**: [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (int8, via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx))
- **TTS**: [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (English, `kokoro-en-v0_19`, via sherpa-onnx)
- **VAD**: [Silero VAD](https://github.com/snakers4/silero-vad) (via sherpa-onnx)

All three run fully offline once downloaded.

## Usage

```python
from parrot import listen, speak

text = listen()      # record from the mic, return the transcript
speak("hello there")  # synthesize and play audio
```

## Setup

```bash
uv sync
```

Download the models into `models/` (gitignored, not shipped with the package):

```bash
cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
tar xf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
mv sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 parakeet
rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2

curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
mv kokoro-en-v0_19 kokoro
rm kokoro-en-v0_19.tar.bz2

curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
mkdir -p silero
mv silero_vad.onnx silero/
```

## Design

Audio is passed around as raw numpy arrays (`(frames, samplerate)`) rather than encoded bytes wherever possible, since mic capture, transcription, and playback all happen in-process on the same device. This skips unnecessary encode/decode overhead, only `audio.py`'s `record_wav`/`play_wav` produce actual WAV bytes, for cases that need portable output (saving to disk, sending over a network).

## Server

Expose `listen`/`speak` over HTTP, for running parrot on one machine (e.g. a Raspberry Pi with a mic and speaker) and calling it from another.

```bash
uv sync --extra server
uv run parrot-serve
```

Binds 0.0.0.0:8420.

- POST /listen — records until a pause in speech, returns {"text": "..."}
- POST /speak — body {"text": "..."}, plays it back, returns {"ok": true}

Both endpoints block until their operation finishes (recording or playback), and access is serialized since there's only one mic and one speaker.