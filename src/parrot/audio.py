"""In-memory audio processing."""

import asciichartpy
import io

import numpy as np
import sounddevice as sd
import soundfile as sf

from collections import deque
from typing import Iterator


# --- raw arrays: for live, in-process use (i.e. streaming to a speech-to-text model) ---

def record(duration: float = 3.0, samplerate: int = 44100, channels: int = 1) -> tuple[np.ndarray, int]:
    """Record from the default mic, returning raw audio for in-process use.

    Args:
        duration: Length of the recording, in seconds.
        samplerate: Sample rate to record at, in Hz.
        channels: Number of input channels (1 = mono, 2 = stereo).
    
    Returns:
        A (frames, samplerate) tuple, where frames is a numpy array of samples.
    """
    frames = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
    sd.wait()
    return frames, samplerate

def play(frames: np.ndarray, samplerate: int) -> None:
    """Play raw audio frames through the default output device.

    Args:
        frames: Numpy array of audio samples.
        samplerate: Sample rate of frames, in Hz.
    
    Returns:
        None. Blocks until playback finishes.
    """
    sd.play(frames, samplerate)
    sd.wait()

def stream_mic(samplerate: int = 16000, block_size: int = 512) -> Iterator[np.ndarray]:
    """Continuously yield raw audio blocks from the default mic.

    Unlike `record`, this doesn't wait for a fixed duration, it opens an
    input stream and yields one block at a time for as long as the caller
    keeps iterating (e.g. until a VAD decides enough silence has passed).

    Args:
        samplerate: Sample rate to record at, in Hz.
        block_size: Number of samples yielded per iteration.

    Yields:
        1-D numpy arrays of `block_size` mono samples, float32.
    """
    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        while True:
            block, _ = stream.read(block_size)
            yield block.reshape(-1)


# --- WAV bytes: for saving to disk, sending over network, etc. ---

def record_wav(duration: float = 3.0, samplerate: int = 44100, channels: int = 1) -> bytes:
    """Record from the default mic and return WAV-encoded bytes.
    
    Args:
        duration: Length of the recording, in seconds.
        samplerate: Sample rate to record at, in Hz.
        channels: Number of input channels (1 = mono, 2 = stereo).
    
    Returns:
        Raw bytes of a WAV file containing the recording.
    """
    # capture audio from the default mic into a numpy array (non-blocking start)
    frames = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels)
    sd.wait()   # block until recording finishes

    # encode the raw samples as a WAV file, in memory
    wav_buf = io.BytesIO()
    wav_buf.name = 'temp.wav'   # soundfile infers format from this
    sf.write(wav_buf, frames, samplerate)
    
    wav_buf.seek(0) # rewind so read() returns all bytes
    return wav_buf.read()

def play_wav(wav: bytes) -> None:
    """Play WAV-encoded audio bytes through the default output device.

    Args:
        wav: Raw bytes of a WAV audio file.
    
    Returns:
        None. Blocks until playback finishes.
    """
    data, samplerate = sf.read(io.BytesIO(wav))
    sd.play(data, samplerate)
    sd.wait()


# --- visualization ---

def plot_mic(samplerate: int = 44100, block_size: int = 1024, width: int = 80, height: int = 15, mode: str = 'waveform') -> None:
    """Continuously plot mic amplitude as a scrolling ASCII waveform until Ctrl+C.

    Args:
        samplerate: Sample rate to capture at, in Hz.
        block_size: Number of samples read per callback invocation.
        width: Number of points shown across the plot (rolling window size).
        height: Height of the plot, in terminal rows.
        mode: 'envelope' (smoothed amplitude level) or 'waveform' (raw signed trace).

    Returns:
        None. Runs until interrupted with Ctrl+C.
    """
    buf = deque([0.0] * width, maxlen=width)    # fixed-size rolling window of amplitude points

    # (indata, frames, time, status) is sounddevice's callback expected signature
    def callback(indata, frames, time, status):
        if status:
            # surface buffer overflows/underflows
            print(status)

        if mode == 'waveform':
            step = max(1, frames // 8)  # sample a few raw points per block, no rectification
            buf.extend(indata[::step, 0].tolist())
        else:
            buf.append(float(np.abs(indata[:, 0]).mean()))  # one averaged point per block

    try:
        with sd.InputStream(samplerate=samplerate, channels=1, blocksize=block_size, callback=callback):
            while True:
                print('\033c', end='')  # clear terminal before redrawing
                print(asciichartpy.plot(list(buf), {'height': height, 'format': '{:8.3f}'}))
                sd.sleep(50)    # ~20 fps redraw
    except KeyboardInterrupt:
        # graceful exit on ^C, no traceback
        print()
