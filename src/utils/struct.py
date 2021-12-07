import inspect


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def to_dict(self):
        return {
            key: value
            for key, value in vars(self).items()
            if not (
                key.startswith('_')
                or inspect.ismethod(key)
            )
        }
