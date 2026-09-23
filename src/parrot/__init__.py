"""Primary functions."""

from contextlib import closing

from parrot.audio import play, stream_mic
from parrot.tts import load_synthesizer, synthesize
from parrot.stt import load_recognizer, load_vad, transcribe


# --- speech-to-text ---

def listen(min_silence_duration: float = 0.5, samplerate: int = 16000) -> str:
    """Record from the mic until a pause in speech, then return its transcript.

    Waits for the VAD to see speech followed by `min_silence_duration` seconds
    of silence, instead of recording a fixed duration, so short utterances
    return quickly and long ones aren't cut off.

    Args:
        min_silence_duration: Length of silence, in seconds, that marks the end of speech.
        samplerate: Sample rate to record at, in Hz (NVIDIA's Parakeet model expects 16kHz).

    Returns:
        The transcribed text of the first detected speech segment.
    """
    recognizer = load_recognizer()
    vad = load_vad(samplerate=samplerate, min_silence_duration=min_silence_duration)
    window_size = vad.config.silero_vad.window_size  # samples the VAD requires per accept_waveform call

    # release the mic as soon as speech ends, rather than whenever the generator gets garbage collected
    with closing(stream_mic(samplerate=samplerate, block_size=window_size)) as mic:
        while vad.empty():
            vad.accept_waveform(next(mic))
    samples = vad.front.samples
    vad.pop()
    return transcribe(recognizer, samples, samplerate)


# --- text-to-speech ---

def speak(text: str) -> None:
    """Synthesize text to audio and play it.
    
    Args:
        text: The text to speak.
    
    Returns:
        None. Blocks until playback finishes.
    """
    synthesizer = load_synthesizer()
    frames, samplerate = synthesize(synthesizer, text)
    play(frames, samplerate)
