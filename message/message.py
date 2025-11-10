import os
from enum import Enum
from pathlib import Path
from typing import Type

_DEFAULT_ERROR: str = "Error"
_SYMBOL_INSERTION_ERROR : str = "Can't do the insertion: no symbols are provided"

class nMessageType(Enum):
    pass

class nMessage:
    _insertion_symbols : str
    _extention : str
    _folder : str
    _command_dict : dict[nMessageType, str]

    def __init__(self, insertion_symbols : str | None, extention : str, folder : str, enum_class : Type[Enum]) -> None:
        self._insertion_symbols = insertion_symbols
        self._extention = extention
        self._folder = folder
        self._command_dict = self._init_command_dict(enum_class=enum_class)

    def _read_file(self, full_filename: str) -> str:
        _output: str = _DEFAULT_ERROR
        try:
            with open(full_filename, 'r', encoding='utf-8') as file:
                _output = file.read()
        except Exception as e:
            raise e

        return _output

    def _get_full_filename(self, type: nMessageType) -> str:
        type_value: str = type.value
        has_subpath: bool = any(sep in type_value for sep in [os.sep, '/'])
        raw_path : str = ""
        if has_subpath:
            raw_path = f"{type_value}.{self._extention}"
        else:
            raw_path: str = f"./{self._folder}/{type_value}.{self._extention}"

        return str(Path(raw_path).as_posix())

    def _insert_text(self, message : str, text_list : list[str]) -> str:
        if not self._insertion_symbols:
            raise Exception(_SYMBOL_INSERTION_ERROR)

        _result : str = _DEFAULT_ERROR
        if not message or len(text_list) == 0 : return _result

        _result = message
        try:
            for text in text_list:
                _result = _result.replace(self._insertion_symbols, text, 1)
        except Exception:
            _result = _DEFAULT_ERROR

        if self._insertion_symbols in _result:
            _result = _result.replace(self._insertion_symbols, "")

        return _result

    def _init_command_dict(self, enum_class: Type[Enum]) -> dict[str, str]:
        _command_dict: dict = {}

        for message_type in enum_class:
            _key: str = message_type.value
            full_filename: str = self._get_full_filename(type=message_type)
            _value: str = self._read_file(full_filename=full_filename)
            _command_dict[_key] = _value

        return _command_dict

    def get_message(self, type : nMessageType) -> str:
        _message : str | None = self._command_dict.get(type.value)
        if _message: return _message

        full_filename : str = self._get_full_filename(type=type)
        _message = self._read_file(full_filename=full_filename)

        self._command_dict[type.value] = _message
        return _message

    def get_modefied_text(self, type : nMessageType, text_list : list[str]) -> str:
        _message : str | None = self._command_dict.get(type.value)
        _modefied_text : str = ""

        if _message:
            _modefied_text : str = self._insert_text(message=_message, text_list=text_list)
        else:
            full_filename : str = self._get_full_filename(type=type)
            text_from_file : str = self._read_file(full_filename=full_filename)
            _modefied_text : str = self._insert_text(message=text_from_file, text_list=text_list)

        return _modefied_text

