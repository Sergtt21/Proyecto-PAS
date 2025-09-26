import json
import redis
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# Definición del Evento
@dataclass
class Event:
    kind: str
    payload: Dict[str, Any]

# Conexión a Redis
redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True 
)

QUEUE = "event_queue"

# API del bus
def put(event: Event):
    data = json.dumps(asdict(event))
    redis_client.rpush(QUEUE, data)


def get(block: bool = True, timeout: int = 5) -> Optional[Event]:
    if block:
        item = redis_client.blpop(QUEUE, timeout=timeout)
        if item:
            _, data = item
            return Event(**json.loads(data))
        return None
    else:
        data = redis_client.lpop(QUEUE)
        if data:
            return Event(**json.loads(data))
        return None
