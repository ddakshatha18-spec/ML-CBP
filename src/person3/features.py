"""Longitudinal feature calculations used by Person 3."""

from typing import Optional


def change(
    current: Optional[float],
    previous: Optional[float],
) -> Optional[float]:
    if current is None or previous is None:
        return None

    return current - previous


def percent_change(
    current: Optional[float],
    previous: Optional[float],
) -> Optional[float]:
    if current is None or previous is None or previous == 0:
        return None

    return ((current - previous) / previous) * 100.0


def rate(
    current: Optional[float],
    previous: Optional[float],
    current_day: float,
    previous_day: float,
) -> Optional[float]:
    if current is None or previous is None:
        return None

    delta_days = current_day - previous_day

    if delta_days == 0:
        return None

    return (current - previous) / delta_days
