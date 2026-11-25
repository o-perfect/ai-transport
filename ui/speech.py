"""Voice input: speech-to-text with Whisper (faster-whisper), run locally.

Free, no API key, works offline once the model is downloaded. The model
("small", ~460 MB, set WHISPER_MODEL to change it) comes from Hugging Face on
first use and is cached in ~/.cache/huggingface. Language is detected
automatically (English, German, Chinese, ...).

If faster-whisper isn't installed, AVAILABLE is False and the chat just hides
the microphone button; nothing else depends on this module.
"""
import io
import os
import threading
import time
import wave
from functools import lru_cache

import numpy as np

from agent.network import CHOKEPOINTS, PORTS

try:
    from faster_whisper import WhisperModel
    AVAILABLE = True
except ImportError:  # optional dependency
    AVAILABLE = False

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

# Whisper's prompt nudges it towards spelling these right (chokepoints, hub ports, jargon).
VOCABULARY = ", ".join([*CHOKEPOINTS, *PORTS, "Red Sea", "Houthi", "TEU", "VLCC", "LNG", "war-risk insurance"])

_lock = threading.Lock()
_warming = False


@lru_cache(maxsize=1)
def _load():
    return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")


def model():
    with _lock:  # the warm-up thread and a first recording may both ask for it
        return _load()


def warm_up():
    """Load the model in the background once per server, so the first
    recording doesn't wait for it."""
    global _warming
    if AVAILABLE and not _warming:
        _warming = True
        threading.Thread(target=model, daemon=True).start()


def _samples(wav: bytes) -> np.ndarray:
    """16 kHz mono float32 samples from a 16-bit PCM WAV (what the browser sends).

    Decoded here with the standard library rather than faster-whisper's own
    decoder, which breaks with PyAV 19 ("unexpected keyword 'metadata_errors'")."""
    with wave.open(io.BytesIO(wav)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM audio, got {8 * width}-bit")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:  # linear resampling is plenty for speech
        n = int(len(audio) * 16000 / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def transcribe(wav: bytes) -> dict:
    """{"text", "language", "seconds"} for a recorded WAV."""
    t = time.time()
    segments, info = model().transcribe(_samples(wav), beam_size=5, vad_filter=True,
                                        initial_prompt=VOCABULARY)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text": text, "language": info.language, "seconds": round(time.time() - t, 1)}
