from telegram import Update, CallbackQuery
from telegram.ext import ContextTypes
from bot.wrappers import chat_wrapper, message_wrapper

async def get_tg_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.callback_query == None:
        return update.message.from_user.id
    query: CallbackQuery = update.callback_query
    await query.answer()
    return query.from_user.id

async def _get_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> CallbackQuery | None:
    if update.callback_query is None: return None
    query: CallbackQuery = update.callback_query
    await query.answer()
    return query

async def send_message_tg(chat_w : chat_wrapper, message_w : message_wrapper) -> None:
    update : Update = chat_w.get_update(); context: ContextTypes.DEFAULT_TYPE= chat_w.get_context()
    text, reply_markup, parse_mode = message_w.unpack()
    query: CallbackQuery | None = await _get_query(update, context)
    if query:
        await query.message.reply_text(text=text, reply_markup = reply_markup, parse_mode=parse_mode)
        return
    await update.message.reply_text(text=text, reply_markup=reply_markup, parse_mode=parse_mode)