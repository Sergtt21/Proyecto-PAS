from queue import Queue
from dataclasses import dataclass

@dataclass
class Event:
    kind: str         # p.ej. "SEND_TEXT"
    payload: dict     # p.ej. {"text": "hola"}

bus = Queue()
