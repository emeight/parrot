"""In-memory audio processing."""

import asciichartpy
import io
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from collections import deque
from collections.abc import Generator
from os import getenv


# --- device handling ---

def _env_device(env_var: str) -> int | str | None:
    """Read a device override from the environment.

    Args:
        env_var: Name of the environment variable, i.e. AUDIO_INPUT_DEVICE.

    Returns:
        A device index, a substring of a device name as listed by `sd.query_devices()`
        (preferred, since indices shift as devices come and go), or None for the system default.
    """
    value = getenv(env_var, "").strip()
    return int(value) if value.isdigit() else (value or None)

# every stream below opens these unless given an explicit device
sd.default.device = (_env_device("AUDIO_INPUT_DEVICE"), _env_device("AUDIO_OUTPUT_DEVICE"))

def open_stream[S: sd.InputStream | sd.OutputStream](
        stream_class: type[S], attempts: int = 4, backoff: float = 0.5, **kwargs
) -> S:
    """Open and start a stream, retrying on PortAudio errors.

    Opening a device can fail transiently (e.g. while the OS is reconfiguring it),
    so each failure is logged, waits a little longer, re-scans devices, and retries.

    Args:
        stream_class: `sd.InputStream` or `sd.OutputStream`.
        attempts: Total number of tries before giving up.
        backoff: Seconds to wait after the first failure, growing linearly per attempt.
        **kwargs: Passed through to `stream_class`.

    Returns:
        A started stream, which the caller is responsible for closing.

    Raises:
        sd.PortAudioError: If every attempt fails.
    """
    for attempt in range(1, attempts):
        try:
            return _start_stream(stream_class, **kwargs)
        except sd.PortAudioError as e:
            print(f"parrot: {e}, retrying ({attempt}/{attempts - 1})", file=sys.stderr)
            time.sleep(backoff * attempt)
            sd._terminate()     # re-initialize PortAudio so it re-scans devices (it only does so at startup)
            sd._initialize()
    return _start_stream(stream_class, **kwargs)

def _start_stream[S: sd.InputStream | sd.OutputStream](stream_class: type[S], **kwargs) -> S:
    """Open and start a single stream, closing it if it fails to start."""
    stream = stream_class(**kwargs)
    try:
        stream.start()
    except sd.PortAudioError:
        stream.close(ignore_errors=True)    # an opened-but-unstarted stream still holds the device
        raise
    return stream

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
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 1:
        frames = frames[:, np.newaxis]  # (samples,) -> (samples, channels)
    with open_stream(sd.OutputStream, samplerate=samplerate, channels=frames.shape[1], dtype="float32") as stream:
        stream.write(frames)    # stream.__exit__ then blocks until the buffer has drained


def stream_mic(samplerate: int = 16000, block_size: int = 512) -> Generator[np.ndarray, None, None]:
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
    with open_stream(sd.InputStream, samplerate=samplerate, channels=1, dtype="float32") as stream:
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
    play(data, samplerate)


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
