from bot.bot import BotHandler
from parameters.parameters import dec_init_parameters, PARAMETERS
from core.logic import COMMANDS

botHandler : BotHandler

@dec_init_parameters()
def main() -> None:

    try:
        botHandler = BotHandler(PARAMETERS['BOT_TOKEN'])
        botHandler.add_command_handlers(COMMANDS)
        botHandler.start_bot()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()