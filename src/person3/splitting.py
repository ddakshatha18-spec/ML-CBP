"""Leakage-safe group splitting helpers."""

from typing import Iterable, List, Tuple, TypeVar

T = TypeVar("T")


def split_by_group(
    items: Iterable[T],
    group_ids: Iterable[str],
    test_fraction: float = 0.2,
) -> Tuple[List[T], List[T]]:
    """Keep every item from the same group in the same split."""

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    pairs = list(zip(items, group_ids))

    groups = []
    seen = set()

    for _, group in pairs:
        if group not in seen:
            seen.add(group)
            groups.append(group)

    test_count = max(1, round(len(groups) * test_fraction)) if groups else 0
    test_groups = set(groups[-test_count:])

    train = [
        item for item, group in pairs
        if group not in test_groups
    ]

    test = [
        item for item, group in pairs
        if group in test_groups
    ]

    return train, test
