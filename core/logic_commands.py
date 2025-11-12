from telegram import Update
from telegram.ext import ContextTypes

from parameters.parameters import PARAMETERS
from core.logic_const import TEXT_LOADER, PROFILE_STORE, AI_ROUTER
from message.text_types import TextType

from core.store.ai_model_settings import (
    AiLevel,
    AiModelSettings,
    get_ai_level,
    get_default_ai_level,
)

from core.ai_module.__aiModule import AiChatRoles
from core.ai_module.ai_formating import format_response

from bot.wrappers import function_wrapper, message_wrapper, chat_wrapper
from bot.chat import send_message_tg, get_tg_id

async def _send_profile_fail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_message(
                TextType.PROFILE_FAIL
            )
        )
    )

async def _start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER, PROFILE_STORE

    tg_id : int = await get_tg_id(update, context)
    default_value : AiLevel = get_default_ai_level()

    PROFILE_STORE.set_record(
        tg_id,
        AiModelSettings(
            default_value
        )
    )

    possiable_options_string : str = " ".join(
        [option.value for option in AiLevel]
    )

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_modefied_text(
                TextType.START,
                [
                    default_value.value,
                    possiable_options_string
                ]
            )
        )
    )

async def _profile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER, PROFILE_STORE

    try:
        tg_id: int = await get_tg_id(update, context)
        profile : AiModelSettings = PROFILE_STORE.get_record(
            tg_id
        )
    except Exception:
        await _send_profile_fail()
        return

    username : str = f"@{update.effective_user.username}"
    ai_level : str = profile._aiLevel.value

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_modefied_text(
                TextType.PROFILE,
                [
                    username,
                    ai_level
                ]
            )
        )
    )

async def _generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER, PROFILE_STORE, AI_ROUTER

    _word_limit : int = int(PARAMETERS['MODEL_WORD_LIMIT'])
    _user_words_len : int = len(context.args)

    async def _send_usage() -> None:
        await send_message_tg(
            chat_w=chat_wrapper(update, context),
            message_w=message_wrapper(
                TEXT_LOADER.get_modefied_text(
                    TextType.GENERATE_USAGE,
                    [
                        str(_word_limit),
                        str(_user_words_len)
                    ]
                )
            )
        )

    async def _send_fail() -> None:
        await send_message_tg(
            chat_w=chat_wrapper(update, context),
            message_w=message_wrapper(
                TEXT_LOADER.get_message(
                    TextType.GENERATE_FAIL
                )
            )
        )

    def get_messages(user_context : str) -> list[dict[str, str]]:
        return [
            {
                "role" : AiChatRoles.USER.value,
                "content" : user_context
            }
        ]


    if _user_words_len > _word_limit:
        await _send_usage()
        return

    try:
        tg_id: int = await get_tg_id(update, context)
        profile : AiModelSettings = PROFILE_STORE.get_record(
            tg_id
        )
    except Exception:
        await _send_profile_fail()
        return


    ai_level : str = profile._aiLevel.value
    user_context : str = " ".join(context.args)

    try:
        ai_response: str = AI_ROUTER.get_response(ai_level, get_messages(user_context))
        generated_text: str = format_response(ai_response)
    except Exception as e:
        print(e)
        await _send_fail()
        return

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_modefied_text(
                TextType.GENERATE,
                [
                    generated_text
                ]
            )
        )
    )

async def _set_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TEXT_LOADER, PROFILE_STORE

    possiable_options : list[str] = [option.value for option in AiLevel]

    async def _send_fail() -> None:
        global TEXT_LOADER

        possiable_options_string : str = " ".join(possiable_options)

        await send_message_tg(
            chat_w=chat_wrapper(update, context),
            message_w=message_wrapper(
                TEXT_LOADER.get_modefied_text(
                    TextType.SET_MODEL_USAGE,
                    [
                        possiable_options_string
                    ]
                )
            )
        )

    if not context.args or len(context.args) > 1:
        await _send_fail()
        return

    user_ai_level : str = context.args[0]
    if user_ai_level not in possiable_options:
        await _send_fail()
        return

    try:
        tg_id : int = await get_tg_id(update, context)
        profile : AiModelSettings = AiModelSettings(
            get_ai_level(
                user_ai_level
            )
        )
        PROFILE_STORE.set_record(
            tg_id,
            profile
        )
    except Exception:
        await _send_fail()
        return

    await send_message_tg(
        chat_w=chat_wrapper(update, context),
        message_w=message_wrapper(
            TEXT_LOADER.get_modefied_text(
                TextType.SET_MODEL_SUCCESS,
                [
                    user_ai_level
                ]
            )
        )
    )


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
            _start
        ),
        function_wrapper(
            TextType.PROFILE.value,
            _profile
        ),
        function_wrapper(
            TextType.GENERATE.value,
            _generate
        ),
        function_wrapper(
            TextType.SET_MODEL.value,
            _set_model
        ),
        function_wrapper(
            TextType.TEST.value,
            _no_implementation
        )
    ]