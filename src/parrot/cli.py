"""Console script entry points."""

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
