"""HTTP server exposing listen/speak over the network."""

import threading

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from parrot import listen, speak

app = FastAPI()
_lock = threading.Lock()  # one mic, one speaker: serialize access


class SpeakRequest(BaseModel):
    text: str


@app.post("/listen")
def listen_endpoint() -> dict[str, str]:
    with _lock:
        return {"text": listen()}


@app.post("/speak")
def speak_endpoint(request: SpeakRequest) -> dict[str, bool]:
    with _lock:
        speak(request.text)
    return {"ok": True}


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8420)


if __name__ == "__main__":
    main()