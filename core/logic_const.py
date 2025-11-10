from message.message import nMessage
from message.text_types import TextType
from parameters.parameters import dec_init_parameters, PARAMETERS
from core.store.profile_store import ProfileStore
from core.store.ai_model_settings import AiModelSettings, AiLevel

@dec_init_parameters()
def _init_text_loader() -> nMessage:
    return nMessage(
        PARAMETERS['INSERTION_SYMBOLS'],
        PARAMETERS['GENERAL_TEMPLATE_EXTENTION'],
        PARAMETERS['GENERAL_TEMPLATE_FOLDER'],
        TextType
    )

@dec_init_parameters()
def _init_profile_store() -> ProfileStore:
    return ProfileStore(
        AiModelSettings(
            AiLevel.ONE
        )
    )

TEXT_LOADER : nMessage = _init_text_loader()
PROFILE_STORE : ProfileStore = _init_profile_store()