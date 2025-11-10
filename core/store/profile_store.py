

class ProfileBase:
    pass

class ProfileStore:
    _store : dict[int, ProfileBase]
    _default_value : ProfileBase | None

    def __init__(self, default_value : ProfileBase) -> None:
        self._store = dict()
        self._default_value = default_value


    def set_record(self, key : int, value : ProfileBase) -> None:

        if not isinstance(key, int):
            raise RuntimeError("Key in profile store should be int")

        if not value:
            value = self._default_value

        self._store[key] = value

    def get_record(self, key : int) -> ProfileBase:

        if not isinstance(key, int):
            raise RuntimeError("Key in profile store should be int")

        record: ProfileBase | None = self._store.get(key)

        if record is None:
            record = self._default_value

        return record


