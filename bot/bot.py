from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ConversationHandler, filters
from telegram import CallbackQuery, Update
from telegram.ext import ContextTypes
from typing import Callable
from bot.wrappers import function_wrapper, conversation_wrapper

class BotHandler:
    _app : Application
    _general_text_dict: dict[str, Callable] = {}
    _query_dict: dict[str, Callable] = {}
    _is_running : bool = False

    def __init__(self, token: str) -> None:
        self._app = Application.builder().token(token).concurrent_updates(True).build()

    def _setup_general_text_handler(self) -> None:
        self._app.add_handler(
            MessageHandler(
                filters=filters.TEXT & ~filters.COMMAND,
                callback=self._handle_general_text
            )
        )

    def _setup_query_handler(self) -> None:
        self._app.add_handler(CallbackQueryHandler(self._handle_query))

    def _setup_handlers(self) -> None:
        self._setup_query_handler()
        self._setup_general_text_handler()

    async def _handle_general_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if context.user_data.get("Conversation"):
            return
        text: str = update.message.text.lower()
        func: Callable | None = self._general_text_dict.get(text)
        if func:
            await func(update, context)

    async def _handle_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query: CallbackQuery | None = update.callback_query
        await query.answer()
        callback_data: str | None = query.data
        func: Callable | None  = self._query_dict.get(callback_data)
        if func:
            await func(update, context)

    def start_bot(self) -> None:
        self._setup_handlers()
        self._app.run_polling(allowed_updates=Update.ALL_TYPES)
        self._is_running = True

    async def stop_bot(self) -> None:
        if not self._is_running: return

        await self._app.stop()
        await self._app.shutdown()
        self._is_running = False

    def load_query_dict(self, query_dict: dict[str, Callable]) -> None:
        self._query_dict = query_dict

    def load_conversations(self, conversations: list[conversation_wrapper]) -> None:
        for conv in conversations:
            self._app.add_handler(
                ConversationHandler(
                    entry_points=conv.enty_points(),
                    states=conv.states(),
                    fallbacks=conv.fallbacks()
                )
            )

    def add_to_general_text(self, general_text_dict: dict[str, Callable]) -> None:
        for key, func in general_text_dict.items():
            self._general_text_dict[key] = func

    def add_command_handlers(self, command_list: list[function_wrapper]) -> None:
        for item in command_list:
            self._app.add_handler(CommandHandler(item.call_back(), item.func()))
