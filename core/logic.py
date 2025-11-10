from bot.wrappers import function_wrapper
from core.logic_commands import get_commands

COMMANDS : list[function_wrapper] = get_commands()