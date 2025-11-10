from enum import Enum
from core.store.profile_store import ProfileBase

class AiLevel(Enum):
    ONE : str = "one"
    TWO : str = "two"

def get_ai_level(value : str) -> AiLevel | None:
    for _v in AiLevel:
        if _v.value == value:
            return _v

    return None

def get_default_ai_level() -> AiLevel:
    arr : list[AiLevel] = [option for option in AiLevel]
    return arr[0]

class AiModelSettings(ProfileBase):
    _aiLevel : AiLevel

    def __init__(self, baseLevel = get_default_ai_level()) -> None:
        self._aiLevel = baseLevel