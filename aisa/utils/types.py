from typing import Union

DICTLISTTYPE = Union[list[str], list[dict[str, any]]]


class dictlist(list):
    def __init__(self, iterable: DICTLISTTYPE):
        super().__init__(iterable)

    def __repr__(self):
        return super().__repr__()
