import numpy as np


def expand_dict(input: dict):
    expanded = [{}]

    for key, values in input.items():
        value_expanded = []

        for obj in expanded:
            value_expanded += __expand_array(key, values, obj)

        expanded = value_expanded

    return expanded


def __expand_array(key: str, values: list, input: dict):
    expanded = []

    if isinstance(values, dict):
        if not all(k in values for k in ("lower", "upper", "num")):
            return expanded

        values = [
            np.random.uniform(values["lower"], values["upper"])
            for _ in range(values["num"])
        ]

    if not isinstance(values, list):
        values = [values]

    for value in values:
        copy = dict(input)
        if value is not None:
            copy[key] = value
        expanded.append(copy)

    return expanded
