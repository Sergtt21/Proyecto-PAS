import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
from bus import get, Event  

# Configuración inicial del bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

if not TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

# Mapa de gestos → texto a enviar
GESTO_TO_TEXT = {
    "DOUBLE_BLINK": "Hola 👋",
    "BROW_UP": "Ya voy 🚗",
}

# Consumidor de eventos
async def _event_consumer(bot: Bot):
    """Escucha eventos persistentes desde Redis y envía mensajes."""
    loop = asyncio.get_running_loop()

    while True:
        try:
            ev: Event = await loop.run_in_executor(None, lambda: get(timeout=5))

            if not ev:
                continue
            #Cada evento deben incluir chat_id
            chat_id = ev.payload.get("chat_id")
            if not chat_id:
                logger.warning("Evento recibido sin 'chat_id'.")
                continue
            #Eventos soportados
            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "")
                if text:
                    try:
                        await bot.send_message(chat_id, text)
                        logger.info(f"📩 Enviado a {chat_id}: {text}")
                    except Exception as e:
                        logger.error(f"❌ No se pudo enviar mensaje a {chat_id}: {e}")

            elif ev.kind == "GESTO":
                name = ev.payload.get("name")
                text = GESTO_TO_TEXT.get(name)
                if text:
                    try:
                        await bot.send_message(chat_id, text)
                        logger.info(f"🤖 Gesto {name} → enviado a {chat_id}")
                    except Exception as e:
                        logger.error(f"❌ No se pudo enviar gesto a {chat_id}: {e}")

        except Exception as e:
            logger.exception(f"❌ Error en _event_consumer: {e}")

# Inicio del bot
async def start_bot():
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )

    dp = Dispatcher()
    @dp.message(Command("start"))
    async def cmd_start(m: Message):
        chat_id = m.chat.id
        logger.info(f"Usuario {chat_id} inició el bot con /start")
        await m.answer(
            f"🤖 Bot activo.\n"
            "Ya puedes recibir notificaciones."
        )

    # /ping → prueba
    @dp.message(Command("ping"))
    async def cmd_ping(m: Message):
        await m.answer("pong ✅")

    # Consumidor de eventos en paralelo
    asyncio.create_task(_event_consumer(bot))

    logger.info("🚀 Bot iniciado. Esperando mensajes...")

    try:
        await dp.start_polling(bot)
    finally:
        logger.info("🛑 Cerrando bot y liberando recursos...")
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot detenido manualmente.")