"""Primary functions."""

from contextlib import closing

from parrot.audio import play, play_remote, stream_mic, stream_remote_mic
from parrot.tts import load_synthesizer, synthesize
from parrot.stt import load_recognizer, load_vad, transcribe


# --- speech-to-text ---

def listen(min_silence_duration: float = 0.5, samplerate: int = 16000, mic_url: str | None = None) -> str:
    """Record from the mic until a pause in speech, then return its transcript.

    Waits for the VAD to see speech followed by `min_silence_duration` seconds
    of silence, instead of recording a fixed duration, so short utterances
    return quickly and long ones aren't cut off.

    Args:
        min_silence_duration: Length of silence, in seconds, that marks the end of speech.
        samplerate: Sample rate to record at, in Hz (NVIDIA's Parakeet model expects 16kHz).
        mic_url: Stream from this remote mic (raw 16kHz PCM) instead of the local one.

    Returns:
        The transcribed text of the first detected speech segment.
    """
    recognizer = load_recognizer()
    vad = load_vad(samplerate=samplerate, min_silence_duration=min_silence_duration)
    window_size = vad.config.silero_vad.window_size  # samples the VAD requires per accept_waveform call

    if mic_url:
        source = stream_remote_mic(mic_url, block_size=window_size)
    else:
        source = stream_mic(samplerate=samplerate, block_size=window_size)

    # release the mic as soon as speech ends, rather than whenever the generator gets garbage collected
    with closing(source) as mic:
        while vad.empty():
            vad.accept_waveform(next(mic))
    samples = vad.front.samples
    vad.pop()
    return transcribe(recognizer, samples, samplerate)


# --- text-to-speech ---

def speak(text: str, speaker_url: str | None = None) -> None:
    """Synthesize text to audio and play it.
    
    Args:
        text: The text to speak.
        speaker_url: Play on this remote speaker instead of the local one.

    Returns:
        None. Blocks until playback finishes.
    """
    synthesizer = load_synthesizer()
    frames, samplerate = synthesize(synthesizer, text)
    if speaker_url:
        play_remote(frames, samplerate, speaker_url)
    else:
        play(frames, samplerate)