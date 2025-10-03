import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv

from .bus import get, Event  # import relativo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

# Mapeo de gestos → mensajes
GESTO_TO_TEXT = {
    "DOUBLE_BLINK": "Hola 👋",
    "BROW_UP": "Ya voy 🚗",
    "SMILE": "Todo bien 😄",
    "NOD": "OK ✅",
    "SHAKE_HEAD": "No ❌"
}

async def _event_consumer(bot: Bot):
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

            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "")
                if text:
                    await bot.send_message(chat_id, text)
                    logger.info(f"📩 Enviado a {chat_id}: {text}")

            elif ev.kind == "GESTO":
                name = ev.payload.get("name")
                text = GESTO_TO_TEXT.get(name)
                if text:
                    await bot.send_message(chat_id, text)
                    logger.info(f"Gesto {name} → enviado a {chat_id} ({text})")
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

    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Cerrando bot y liberando recursos...")
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot detenido manualmente.")
