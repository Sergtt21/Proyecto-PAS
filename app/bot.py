import os
import asyncio
import logging
import time
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from collections import defaultdict
from .bus import get, Event  # import relativo

# Importar el módulo de manejo de logs
from .managelog import manejo_errores
manejo_errores(nivel_warning="ignore", verbose=False) 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Control de rate-limit
_last_sent_time_by_gesture = defaultdict(lambda: 0.0)
_global_last_sent_time = 0.0
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "10.0").strip() or 10.0)
_sent_count = 0
_limited_count = 0

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

# Mapeo de gestos → mensajes
GESTO_TO_TEXT = {
    "DOBLE_PARPADEO":   "Hola, como estas ?👋",
    "CEJAS_ARRIBA":     "Gracias, Hasta pronto🙌",
    "SONRISA":          "Todo bien 😄",
    "CABEZA_DERECHA":   "Listo ✅",
    "CABEZA_IZQUIERDA": "No puedo ❌",
}

async def _event_consumer(bot: Bot):
    global _global_last_sent_time, _sent_count, _limited_count
    loop = asyncio.get_running_loop()
    while True:
        try:
            ev: Event = await loop.run_in_executor(None, lambda: get(timeout=5))
            if not ev:
                continue

            chat_id = ev.payload.get("chat_id")
            if not chat_id:
                logger.warning("Evento recibido sin 'chat_id'.")
                continue

            # Rate-limit GLOBAL: si ha pasado menos de RATE_LIMIT_SECONDS desde el último envío, no enviar nada.
            now = time.time()
            if (now - _global_last_sent_time) < RATE_LIMIT_SECONDS:
                _limited_count += 1
                logger.info(
                    f"Rate-limit GLOBAL: evento {ev.kind} ignorado "
                    f"({RATE_LIMIT_SECONDS - (now - _global_last_sent_time):.1f}s restantes) | Bloqueados: {_limited_count}"
                )
                continue

            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "")
                if text:
                    await bot.send_message(chat_id, text)
                    _global_last_sent_time = time.time()
                    _sent_count += 1
                    logger.info(f"📩 Enviado a {chat_id}: {text} | Total enviados: {_sent_count}")

            elif ev.kind == "GESTO":
                name = ev.payload.get("name")
                text = GESTO_TO_TEXT.get(name)
                if text:
                    # (opcional) rate-limit por gesto además del global
                    elapsed_g = now - _last_sent_time_by_gesture[name]
                    if elapsed_g < RATE_LIMIT_SECONDS:
                        _limited_count += 1
                        logger.info(
                            f"Rate-limit por gesto: {name} ignorado "
                            f"(faltan {RATE_LIMIT_SECONDS - elapsed_g:.1f}s) | Bloqueados: {_limited_count}"
                        )
                        continue

                    await bot.send_message(chat_id, text)
                    _last_sent_time_by_gesture[name] = time.time()
                    _global_last_sent_time = _last_sent_time_by_gesture[name]
                    _sent_count += 1
                    logger.info(f"Gesto {name} → enviado a {chat_id} ({text}) | Total enviados: {_sent_count}")
                else:
                    logger.warning(f"Gesto {name} no reconocido.")

        except Exception:
            logger.exception("Error en _event_consumer")

async def start_bot():
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def cmd_start(m: Message):
        chat_id = m.chat.id
        logger.info(f"ID del chat: {m.chat.id}")
        logger.info(f"Usuario {chat_id} inició el bot con /start")
        await m.answer("Bot activo.\nYa puedes recibir notificaciones de gestos.")

    @dp.message(Command("ping"))
    async def cmd_ping(m: Message):
        await m.answer("pong ✅")

    asyncio.create_task(_event_consumer(bot))
    logger.info("Bot iniciado. Esperando mensajes...")

    while True:
        try:
            await dp.start_polling(bot)
        except Exception as e:
            logger.error(f"Conexión perdida con Telegram: {e}. Reintentando en 5 s...")
            await asyncio.sleep(5)
        else:
            break
    await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot detenido manualmente.")
