"""Speech-to-text (STT) model wrappers."""

import sherpa_onnx

import numpy as np


def load_recognizer(
    encoder: str = "models/parakeet/encoder.int8.onnx",
    decoder: str = "models/parakeet/decoder.int8.onnx",
    joiner: str = "models/parakeet/joiner.int8.onnx",
    tokens: str = "models/parakeet/tokens.txt",
) -> sherpa_onnx.OfflineRecognizer:
    """Load the offline Parakeet TDT recognizer.

    Args:
        encoder: Path to the encoder ONNX model.
        decoder: Path to the decoder ONNX model.
        joiner: Path to the joiner ONNX model.
        tokens: Path to the tokens file.

    Returns:
        A loaded OfflineRecognizer.
    """
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        model_type="nemo_transducer",
        num_threads=4,
    )


def transcribe(recognizer: sherpa_onnx.OfflineRecognizer, frames: np.ndarray, samplerate: int) -> str:
    """Run the recognizer on a chunk of raw audio.

    Args:
        recognizer: A loaded OfflineRecognizer.
        frames: Mono audio samples, as a numpy array or a plain list (sherpa-onnx's VAD segments return samples as a list).
        samplerate: Sample rate of frames, in Hz.

    Returns:
        The transcribed text.
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(samplerate, np.asarray(frames).flatten())
    recognizer.decode_stream(stream)
    return stream.result.text


def load_vad(
    model: str = "models/silero/silero_vad.onnx",
    samplerate: int = 16000,
    min_silence_duration: float = 0.5,
) -> sherpa_onnx.VoiceActivityDetector:
    """Load the Silero voice activity detector.

    Args:
        model: Path to the silero_vad.onnx model.
        samplerate: Sample rate to run the VAD at, in Hz (must match the mic and recognizer).
        min_silence_duration: Length of silence, in seconds, needed to close a speech segment.

    Returns:
        A loaded VoiceActivityDetector.
    """
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = model
    config.silero_vad.min_silence_duration = min_silence_duration
    config.sample_rate = samplerate
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=100)
