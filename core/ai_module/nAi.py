import json
import copy
import random
from enum import Enum
from typing import Final
from httpx import Response, post

MAX_SEED_VALUE: Final[int] = 2**32 - 1
USER_CHAT_TEXT : Final[str] = """
    Сгеренируй, что требуется в системном промпте.
    Запрещено повторять предыдущие ситуации.
    Всегда старайся сгенерировать новый тип, контекст и персонажий
"""
_NO_GENERATION_ERROR : Final[str] = "Response result doesn't contain generated text"

class nAiChatRoles(Enum):
    USER : str = "user"
    SYSTEM : str = "system"
    ASSISTANT : str = "assistant"

class nAi:
    _model : str
    _api_key : str
    _api_url : str
    _post_json : dict
    _timeout : int
    _chat_history_limit : int

    def __init__(
        self,
        model : str,
        api_key : str,
        url : str,
        timeout : int,
        post_json_filepath : str,
        chat_history_limit : int
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_url = url
        self._timeout = int(timeout)
        self._chat_history_limit = int(chat_history_limit)
        self._load_json(post_json_filepath)

    def generate(self, system_message : str, history: list[str] = []) -> str:
        global _NO_GENERATION_ERROR

        resp: Response = post(
            self._api_url,
            headers=self._get_headers(),
            json=self._get_json_content(
                system_message=system_message,
                history=history
            ),
            timeout=self._timeout
        )

        _result : dict = resp.json()
        if _result.get("error"):
            raise RuntimeError(_result.get("error"))
        if not _result.get("choices"):
            raise Exception(_NO_GENERATION_ERROR)

        return _result["choices"][0]["message"]["content"]

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }

    def _get_json_content(self, system_message : str, history : list[str]) -> dict:
        _result_post_json : dict = copy.deepcopy(self._post_json)
        _result_post_json["model"] = self._model

        _result_post_json = self._insert_messages(
            _result_post_json,
            system_message,
            history
        )
        _result_post_json["seed"] = self._get_seed()
        return _result_post_json

    def _insert_messages(self, post_json : dict, system_message : str, history: list[str]) -> dict:
        global USER_CHAT_TEXT

        _messages_property : list[dict[str,str]] = post_json["messages"]

        _system_message : dict[str, str] = {
            "role": nAiChatRoles.SYSTEM.value,
            "content": system_message
        }

        _user_message : dict[str, str] = {
            "role" : nAiChatRoles.USER.value,
            "content" : USER_CHAT_TEXT
        }

        _assistent_message : dict[str, str] = {
            "role" : nAiChatRoles.ASSISTANT.value,
            "content" : ""
        }

        _messages_property.append(_system_message)

        _len : int = min(self._chat_history_limit, len(history))

        for id in range(_len):
            _messages_property.append(_user_message)
            _assistent_message["content"] = str(history[id])
            _messages_property.append(_assistent_message)

        _messages_property.append(_user_message)

        return post_json

    def _load_json(self, filename : str) -> None:
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self._post_json = json.load(file)
        except Exception as e:
            self._post_json = {}

    def _get_seed(self) -> int:
        global MAX_SEED_VALUE
        return random.randint(0, MAX_SEED_VALUE)