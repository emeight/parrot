"""Console script entry points."""
import os

from parrot import listen, speak


def repeat() -> None:
    """Record until a pause, then repeat back what was heard.

    Returns:
        None.
    """
    text = listen()
    print(text)
    speak(text)


def scribe() -> None:
    """Continuously record until each pause, printing every transcript.

    Returns:
        None. Runs until interrupted with Ctrl+C.
    """
    try:
        while True:
            print(listen())
    except KeyboardInterrupt:
        print()


def remote() -> None:
    """Repeat back what's heard, using m0pi's mic and speaker at M0PI_URL.

    Returns:
        None. Runs until interrupted with Ctrl+C.
    """
    url = os.environ["M0PI_URL"]
    try:
        while True:
            text = listen(mic_url=f"{url}/mic")
            print(text)
            if text:    # the VAD can fire on noise, and that transcribes to nothing
                speak(text, speaker_url=f"{url}/speaker")
    except KeyboardInterrupt:
        print()
