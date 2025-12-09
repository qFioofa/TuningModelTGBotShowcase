from typing import Final
from core.ai_module.nAi import nAi
from core.ai_module.model_app_2 import TUNED_MODEL
from core.store.ai_model_settings import AiLevel, ai_level_to_model

_DEFAULT_SYSTEM_PROMT : Final[str] = """
    Сгенерируй тип текста, который ты лучше всего умеешь.
"""

class AiRouter:
    _rout : dict[AiLevel , nAi]
    _system_message : str | None

    def __init__(self,
        api_key : str,
        url : str,
        timeout : int,
        post_json_filepath : str,
        chat_history_limit : int
    ) -> None:
        self._rout = {}
        self._init_router(
            api_key,
            url,
            timeout,
            post_json_filepath,
            chat_history_limit,
        )

    def _init_router(self,
        api_key : str,
        url : str,
        timeout : int,
        post_json_filepath : str,
        chat_history_limit : int
    ) -> None:
        for level in AiLevel:
            _base_model_name : str = ai_level_to_model(level)

            try:
                self._rout[level] = nAi(
                    _base_model_name,
                    api_key,
                    url,
                    timeout,
                    post_json_filepath,
                    chat_history_limit,
                )
            except Exception:
                pass

    def init_system_message(self, system_message : str) -> None:
        self._system_message = system_message

    def get_response(self, model : AiLevel, user_response : str) -> str:
        return TUNED_MODEL.get_response(user_response)
        # _model : nAi | None = self._rout.get(model)
        #
        # if _model is None:
        #     raise RuntimeError(f"Model: {model.value} does not exist or failed to load")
        #
        # response : str = _model.generate(
        #     self._system_message,
        #     [
        #         user_response
        #     ]
        # )
        #
        # return response
        # _model : nAi | None = self._rout.get(model)
        #
        # if _model is None:
        #     raise RuntimeError(f"Model: {model.value} does not exist or failed to load")
        #
        # if self._system_message is None:
        #     global _DEFAULT_SYSTEM_PROMT
        #     self._system_message = _DEFAULT_SYSTEM_PROMT
        #
        # response : str = _model.generate(
        #     self._system_message,
        #     user_response
        # )
        #
        # return response
