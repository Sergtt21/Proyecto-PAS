import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.filters import Command
from dotenv import load_dotenv

from .bus import bus

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_CHAT_ID = int(os.getenv("DEFAULT_CHAT_ID", "0"))

bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    await msg.answer("Bot listo. Enviaré mensajes cuando detectes el gesto configurado.")

@dp.message(Command("ping"))
async def cmd_ping(msg: types.Message):
    await msg.answer("pong ✅")

async def event_consumer():
    """Lee eventos del bus y actúa (enviar textos, etc.)."""
    while True:
        ev = await asyncio.get_event_loop().run_in_executor(None, bus.get)
        try:
            if ev.kind == "SEND_TEXT":
                text = ev.payload.get("text", "Hola")
                chat_id = ev.payload.get("chat_id", DEFAULT_CHAT_ID)
                if chat_id:
                    await bot.send_message(chat_id, text)
            # agrega otros tipos: SNAPSHOT, ALERT, etc.
        finally:
            bus.task_done()

async def start_bot():
    # Correr consumidor en paralelo
    asyncio.create_task(event_consumer())
    # Arrancar polling
    await dp.start_polling(bot)
