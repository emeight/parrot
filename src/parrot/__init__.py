"""Primary functions."""

from parrot.audio import record, play
from parrot.tts import load_synthesizer, synthesize
from parrot.stt import load_recognizer, transcribe


# --- speech-to-text ---

def listen(chunk_duration: float = 4.0, samplerate: int = 16000) -> str:
    """Record one chunk from the mic and return its transcript.
    
    Args:
        chunk_duration: Length of the recording, in seconds.
        samplerate: Sample rate to record at, in Hz (NVIDIA's Parakeet model expects 16kHz).
    
    Returns:
        The transcribed text.
    """
    recognizer = load_recognizer()
    frames, sr = record(duration=chunk_duration, samplerate=samplerate)
    return transcribe(recognizer, frames, sr)


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
