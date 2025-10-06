from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

import redis
from dotenv import load_dotenv

load_dotenv()

<<<<<<< HEAD
# Configuración 
=======
>>>>>>> master
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_HOST = os.getenv("REDIS_HOST", "localhost").strip()
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
BUS_QUEUE = os.getenv("BUS_QUEUE", "event_queue").strip()

<<<<<<< HEAD
# Evento 
=======
>>>>>>> master
@dataclass
class Event:
    kind: str
    payload: Dict[str, Any]
<<<<<<< HEAD
    ts: float = field(default_factory=time.time)  # marca de tiempo

# Cliente Redis 
def _make_client() -> redis.Redis:
    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True
    )

_client = _make_client()

# API 
def put(event: Event) -> None:
    """Encola un evento en Redis (FIFO)."""
    data = json.dumps(asdict(event), ensure_ascii=False)
    try:
        _client.rpush(BUS_QUEUE, data)
    except redis.exceptions.RedisError as e:
        raise RuntimeError(f"Redis put() falló: {e}") from e

def get(block: bool = True, timeout: int = 5) -> Optional[Event]: # block=True: espera hasta timeout (S). block=False: intenta leer inmediato.  Devuelve None si no hay evento disponible.
    try:
        if block:
            item = _client.blpop(BUS_QUEUE, timeout=timeout)
            if not item:
                return None
            _, data = item
        else:
            data = _client.lpop(BUS_QUEUE)
            if data is None:
                return None

        obj = json.loads(data)
        if "ts" not in obj: # compatibilidad hacia atrás
            obj["ts"] = time.time()
        return Event(**obj)

    except redis.exceptions.RedisError:
        # en caso de error de conexión devolvemos None para que el loop siga vivo
        return None
=======
    ts: float = field(default_factory=time.time)

def _make_client() -> redis.Redis:
    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

_client = _make_client()

def put(event: Event) -> None:
    data = json.dumps(asdict(event), ensure_ascii=False)
    _client.rpush(BUS_QUEUE, data)

def get(block: bool = True, timeout: int = 5) -> Optional[Event]:
    if block:
        item = _client.blpop(BUS_QUEUE, timeout=timeout)
        if not item:
            return None
        _, data = item
    else:
        data = _client.lpop(BUS_QUEUE)
        if data is None:
            return None

    obj = json.loads(data)
    if "ts" not in obj:
        obj["ts"] = time.time()
    return Event(**obj)
>>>>>>> master
