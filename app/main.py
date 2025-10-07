import threading
import asyncio
from app.vision import start_gesture_detection
from app.bot import start_bot

def run_vision():
    start_gesture_detection()

def run_bot():
    asyncio.run(start_bot())

if __name__ == "__main__":
    t = threading.Thread(target=run_vision, daemon=True)
    t.start()
    run_bot()
