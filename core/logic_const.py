from message.message import nMessage
from message.text_types import TextType
from parameters.parameters import dec_init_parameters, PARAMETERS

@dec_init_parameters()
def _init_text_loader() -> nMessage:
    return nMessage(
        PARAMETERS['INSERTION_SYMBOLS'],
        PARAMETERS['GENERAL_TEMPLATE_EXTENTION'],
        PARAMETERS['GENERAL_TEMPLATE_FOLDER'],
        TextType
    )

TEXT_LOADER : nMessage = _init_text_loader()