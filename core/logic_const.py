from message.message import nMessage
from message.text_types import TextType

from parameters.parameters import dec_init_parameters, PARAMETERS

from core.store.profile_store import ProfileStore
from core.store.ai_model_settings import AiModelSettings, AiLevel

from core.ai_module.AiRouter import AiRouter
from core.ai_module.nAi import nAi


@dec_init_parameters()
def _init_text_loader() -> nMessage:
    global PARAMETERS

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

@dec_init_parameters()
def _init_ai_module() -> AiRouter:
    global PARAMETERS

    return AiRouter(
        PARAMETERS['AI_TOKEN'],
        PARAMETERS['AI_URL'],
        PARAMETERS['AI_RESPONSE_TIMEOUT'],
        PARAMETERS['AI_POST_JSON'],
        PARAMETERS['AI_HISTORY_CHAT_LIMIT']
    )

TEXT_LOADER : nMessage = _init_text_loader()
PROFILE_STORE : ProfileStore = _init_profile_store()
AI_ROUTER : AiRouter = _init_ai_module()