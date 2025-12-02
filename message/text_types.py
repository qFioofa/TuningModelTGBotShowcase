from message.message import nMessageType

class TextType(nMessageType):
    START : str = "start"
    GENERATE : str = "generate"
    SET_MODEL : str = "set_model"
    PROFILE : str = "profile"

    NO_IMPLEMENTATION : str = "no_implementation"

    GENERATE_USAGE : str = "generate_usage"
    SET_MODEL_USAGE : str = "set_model_usage"

    GENERATE_PROCCESS : str = "generate_proccess"
    SYSTEM_PROMT : str = "system_promt"

    SET_MODEL_SUCCESS : str = "set_model_success"

    PROFILE_FAIL : str = "profile_fail"
    GENERATE_FAIL : str = "generate_fail"

    TEST : str = "test"