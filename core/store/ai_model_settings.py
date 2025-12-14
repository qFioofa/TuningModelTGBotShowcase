from enum import Enum
from core.store.profile_store import ProfileBase

class AiModelNames(Enum):
    ONE: str = "BroneBonBon/Conflict-Generator-Mistral-v2"
    phi : str = "BroneBonBon/Conflict-Generator-Phi-v2"

class AiLevel(Enum):
    ONE : str = "conflict_mistral"
    TWO : str = "conflict_phi"

_MODEL_TO_LEVEL_DICT : dict[AiLevel, AiModelNames] = {
    AiLevel.ONE : AiModelNames.ONE,
    AiLevel.TWO : AiModelNames.phi
}

def ai_level_to_model(level : AiLevel) -> str | None:
    global _MODEL_TO_LEVEL_DICT

    return _MODEL_TO_LEVEL_DICT.get(
        level
    ).value

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
