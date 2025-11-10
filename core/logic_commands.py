from telegram import Update
from telegram.ext import ContextTypes

from core.logic_const import TEXT_LOADER
from message.text_types import TextType
from bot.wrappers import function_wrapper, message_wrapper, chat_wrapper
from bot.chat import send_message_tg

async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_message(
                TextType.START
            )
        )
    )

async def _generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pass

async def _set_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pass


async def _no_implementation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_message(
                TextType.NO_IMPLEMENTATION
            )
        )
    )

def get_commands() -> list[function_wrapper]:
    return [
        function_wrapper(
            TextType.START.value,
            _no_implementation
        ),
        function_wrapper(
            TextType.GENERATE.value,
            _no_implementation
        ),
        function_wrapper(
            TextType.SET_MODEL.value,
            _no_implementation
        )
    ]