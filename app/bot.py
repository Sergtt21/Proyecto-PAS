import os
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
<<<<<<< HEAD
from bus import get, Event  

# Configuración inicial del bot
=======

from .bus import get, Event  #import relativo
>>>>>>> master

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
<<<<<<< HEAD

if not TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

# Mapa de gestos → texto a enviar
=======
if not TOKEN:
    raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

>>>>>>> master
GESTO_TO_TEXT = {
    "DOUBLE_BLINK": "Hola 👋",
    "BROW_UP": "Ya voy 🚗",
}

<<<<<<< HEAD
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
=======
async def _event_consumer(bot: Bot):
    loop = asyncio.get_running_loop()
    while True:
        try:
            ev: Event = await loop.run_in_executor(None, lambda: get(timeout=5))
            if not ev:
                continue

>>>>>>> master
            chat_id = ev.payload.get("chat_id")
            if not chat_id:
                logger.warning("Evento recibido sin 'chat_id'.")
                continue
<<<<<<< HEAD
            #Eventos soportados
            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "")
                if text:
                    try:
                        await bot.send_message(chat_id, text)
                        logger.info(f"📩 Enviado a {chat_id}: {text}")
                    except Exception as e:
                        logger.error(f"❌ No se pudo enviar mensaje a {chat_id}: {e}")
=======

            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "")
                if text:
                    await bot.send_message(chat_id, text)
                    logger.info(f"📩 Enviado a {chat_id}: {text}")
>>>>>>> master

            elif ev.kind == "GESTO":
                name = ev.payload.get("name")
                text = GESTO_TO_TEXT.get(name)
                if text:
<<<<<<< HEAD
                    try:
                        await bot.send_message(chat_id, text)
                        logger.info(f"🤖 Gesto {name} → enviado a {chat_id}")
                    except Exception as e:
                        logger.error(f"❌ No se pudo enviar gesto a {chat_id}: {e}")

        except Exception as e:
            logger.exception(f"❌ Error en _event_consumer: {e}")

# Inicio del bot
=======
                    await bot.send_message(chat_id, text)
                    logger.info(f"Gesto {name} → enviado a {chat_id}")

        except Exception:
            logger.exception("Error en _event_consumer")

>>>>>>> master
async def start_bot():
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
<<<<<<< HEAD

    dp = Dispatcher()
=======
    dp = Dispatcher()

>>>>>>> master
    @dp.message(Command("start"))
    async def cmd_start(m: Message):
        chat_id = m.chat.id
        logger.info(f"Usuario {chat_id} inició el bot con /start")
<<<<<<< HEAD
        await m.answer(
            f"🤖 Bot activo.\n"
            "Ya puedes recibir notificaciones."
        )

    # /ping → prueba
=======
        await m.answer("Bot activo.\nYa puedes recibir notificaciones.")

>>>>>>> master
    @dp.message(Command("ping"))
    async def cmd_ping(m: Message):
        await m.answer("pong ✅")

<<<<<<< HEAD
    # Consumidor de eventos en paralelo
    asyncio.create_task(_event_consumer(bot))

    logger.info("🚀 Bot iniciado. Esperando mensajes...")
=======
    asyncio.create_task(_event_consumer(bot))
    logger.info("Bot iniciado. Esperando mensajes...")
>>>>>>> master

    try:
        await dp.start_polling(bot)
    finally:
<<<<<<< HEAD
        logger.info("🛑 Cerrando bot y liberando recursos...")
        await bot.session.close()


=======
        logger.info("Cerrando bot y liberando recursos...")
        await bot.session.close()

>>>>>>> master
if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
<<<<<<< HEAD
        logger.info("🛑 Bot detenido manualmente.")
=======
        logger.info("Bot detenido manualmente.")
>>>>>>> master
