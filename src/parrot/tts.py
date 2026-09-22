"""Text-to-speech (TTS) model wrappers."""

import sherpa_onnx

import numpy as np


def load_synthesizer(
    model: str = "models/kokoro/model.onnx",
    voices: str = "models/kokoro/voices.bin",
    tokens: str = "models/kokoro/tokens.txt",
    data_dir: str = "models/kokoro/espeak-ng-data",
) -> sherpa_onnx.OfflineTts:
    """Load the offline Kokoro TTS model.
    
    Args:
        model: Path to the Kokoro ONNX model.
        voices: Path to the voices file.
        tokens: Path to the tokens file.
        data_dir: Path to the bundled espeak-ng-data directory.

    Returns:
        A loaded OfflineTts.
    """
    kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=model, voices=voices, tokens=tokens, data_dir=data_dir
    )
    model_config = sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=4)
    config = sherpa_onnx.OfflineTtsConfig(model=model_config)
    return sherpa_onnx.OfflineTts(config)


def synthesize(
    synthesizer: sherpa_onnx.OfflineTts, text: str, speaker_id: int = 10, speed: float = 1.0
) -> tuple[np.ndarray, int]:
    """Synthesize text to raw audio.

    Args:
        synthesizer: A loaded OfflineTts.
        text: The text to synthesize.
        speaker_id: Speaker ID, for multi-speaker models.
        speed: Speaking speed, larger values produce faster speech.

    Returns:
        A (frames, samplerate) tuple, where frames is a numpy array of samples.
    """
    audio = synthesizer.generate(text, sid=speaker_id, speed=speed)
    return audio.samples, audio.sample_rate