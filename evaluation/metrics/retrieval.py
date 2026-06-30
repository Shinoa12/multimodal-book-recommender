from collections.abc import Iterable
from typing import TypeVar


Tag = TypeVar("Tag")


def compute_jaccard(tags_a: Iterable[Tag], tags_b: Iterable[Tag]) -> float:
    """Compute the Jaccard coefficient between two tag collections.

    The Jaccard coefficient is the size of the intersection divided by the
    size of the union. If both collections are empty, the result is 1.0.
    """
    set_a = set(tags_a)
    set_b = set(tags_b)

    union = set_a | set_b
    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)
