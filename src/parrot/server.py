"""HTTP server exposing listen/speak over the network."""

import threading

import sounddevice as sd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from parrot import listen, speak

app = FastAPI()
_lock = threading.Lock()  # one mic, one speaker: serialize access

# ValueError: no device matches AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE
_AUDIO_ERRORS = (sd.PortAudioError, ValueError)


class SpeakRequest(BaseModel):
    text: str


@app.post("/listen")
def listen_endpoint() -> dict[str, str]:
    with _lock:
        try:
            return {"text": listen()}
        except _AUDIO_ERRORS as e:
            raise HTTPException(status_code=503, detail=f"audio device error: {e}") from e


@app.post("/speak")
def speak_endpoint(request: SpeakRequest) -> dict[str, bool]:
    with _lock:
        try:
            speak(request.text)
        except _AUDIO_ERRORS as e:
            raise HTTPException(status_code=503, detail=f"audio device error: {e}") from e
    return {"ok": True}


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8420)


if __name__ == "__main__":
    main()