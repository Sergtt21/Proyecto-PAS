"""
Script de prueba para la cola de eventos (bus.py).

IMPORTANTE:
Cada miembro del equipo debe configurar sus propios
TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en el archivo `.env`.

* TELEGRAM_BOT_TOKEN lo entrega BotFather al crear el bot.
* TELEGRAM_CHAT_ID puede ser el ID personal del usuario o el de un grupo
(asegurarse de haber iniciado el chat con el bot en Telegram).

Prerequisitos para ejecutar la prueba:
1. Tener Redis corriendo (ej: `docker run -p 6379:6379 redis:7`).
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Configurar `.env` con los valores correctos (token y chat_id).
4. Levantar el bot: `python app/bot.py`.
5. En otra terminal, ejecutar este script: `python send_test_event.py`.

Resultado esperado:
    Deberías recibir en Telegram el mensaje: "✅ Prueba BUS → BOT".
"""

import os
from dotenv import load_dotenv
from app.bus import put, Event  # 👈 import desde paquete app

load_dotenv()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
assert CHAT_ID, "⚠️ Falta TELEGRAM_CHAT_ID en el archivo .env"

put(Event(kind="SEND_TEXT", payload={"chat_id": CHAT_ID, "text": "✅ Prueba BUS → BOT"}))
print("Evento publicado ✔️")
