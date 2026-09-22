"""Speech-to-text (STT) model wrappers."""

import sherpa_onnx

import numpy as np

from parrot.audio import record


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
        frames: Numpy array of mono audio samples.
        samplerate: Sample rate of frames, in Hz.

    Returns:
        The transcribed text.
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(samplerate, frames.flatten())
    recognizer.decode_stream(stream)
    return stream.result.text


def live_transcribe(chunk_duration: float = 4.0, samplerate: int = 16000) -> None:
    """Continuously record fixed-length chunks and print their transcripts.

    Args:
        chunk_duration: Length of each chunk, in seconds.
        samplerate: Sample rate to record at, in Hz (Parakeet expects 16kHz).

    Returns:
        None. Runs until interrupted with Ctrl+C.
    """
    recognizer = load_recognizer()
    try:
        while True:
            frames, sr = record(duration=chunk_duration, samplerate=samplerate)
            text = transcribe(recognizer, frames, sr)
            if text:
                print(text)
    except KeyboardInterrupt:
        print()
