import os
from dotenv import load_dotenv
from typing import Callable

PARAMETERS: dict = {}

def _get_parameters() -> dict:
    load_dotenv()
    return dict(os.environ)

def init_parameters() -> None:
    global PARAMETERS

    PARAMETERS.update(_get_parameters())

    if PARAMETERS is None:
        raise Exception("Can't load env parameters")

def dec_init_parameters() -> Callable:
    def decorator(function: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> None:

            if not hasattr(dec_init_parameters, '_inited') or not dec_init_parameters._inited:
                init_parameters()
                dec_init_parameters._inited = True

            return function(*args, **kwargs)

        return wrapper
    return decorator