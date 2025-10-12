# app/main.py
import threading
import asyncio
import signal
import sys

from app.vision import start_gesture_detection
from app.bot import start_bot

def run_vision():
    start_gesture_detection()  # sale con ESC

def run_bot():
    asyncio.run(start_bot())

if __name__ == "__main__":
    try:
        t = threading.Thread(target=run_vision, daemon=True)
        t.start()
        run_bot()
    except KeyboardInterrupt:
        print("\nInterrupción recibida. Cerrando…")
        sys.exit(0)