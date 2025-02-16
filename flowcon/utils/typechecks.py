"""Functions that check types."""

from typing import Any


def is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def is_int(x: Any) -> bool:
    return isinstance(x, int)


def is_positive_int(x: Any) -> bool:
    return is_int(x) and x > 0


def is_nonnegative_int(x: Any) -> bool:
    return is_int(x) and x >= 0


def is_power_of_two(n: int) -> bool:
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False
