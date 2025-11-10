from typing import Callable
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, Update
from telegram.ext import ContextTypes

class function_wrapper:
    _call_back : str
    _function: Callable

    def __init__(self, call_back : str, function_ :  Callable) -> None:
        self._call_back = call_back
        self._function = function_

    def call_back(self)->str:
        return self._call_back

    def func(self)->Callable:
        return self._function

class conversation_wrapper:
    _enty_points: list = []
    _states: dict = {}
    _fallbacks: list = []

    def __init__(self, enty_points: list, states: dict, fallbacks: list = [])->None:
        self._enty_points = enty_points
        self._states = states
        self._fallbacks = fallbacks

    def enty_points(self) -> list:
        return self._enty_points

    def states(self) -> dict:
        return self._states

    def fallbacks(self) -> list:
        return self._fallbacks

class message_wrapper:
    _text : str | None
    _reply_markup: InlineKeyboardMarkup | list[list[InlineKeyboardButton]] | None
    _parse_mode: str | None

    def __init__(
            self,
            text: str,
            replay_markup: InlineKeyboardMarkup | list[list[InlineKeyboardButton]] | None = None,
            parse_mode : str = "Markdown"
        ) -> None:
        self._text = text

        if isinstance(replay_markup, list):
            self._reply_markup = InlineKeyboardMarkup(replay_markup)
        else:
            self._reply_markup = replay_markup

        self._parse_mode = parse_mode

    def get_text(self) -> str:
        return self._text

    def get_replay_markup(self) -> InlineKeyboardMarkup:
        return self._reply_markup

    def get_parse_mode(self) -> str:
        return self._parse_mode

    def unpack(self) -> tuple:
        return self._text, self._reply_markup, self._parse_mode

class chat_wrapper:
    update: Update
    context : ContextTypes.DEFAULT_TYPE

    def __init__(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.update = update
        self.context = context

    def get_update(self) -> Update:
        return self.update

    def get_context(self) -> ContextTypes.DEFAULT_TYPE:
        return self.context

    def unpack(self) -> tuple:
        return self.update, self.context