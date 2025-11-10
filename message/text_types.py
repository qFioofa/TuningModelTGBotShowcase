from message.message import nMessageType

class TextType(nMessageType):
    START : str = "start"
    GENERATE : str = "generate"
    SET_MODEL : str = "set_model"

    GENERATE_USAGE : str = "generate_usage"
    SET_MODEL_USAGE : str = "set_model_usage"