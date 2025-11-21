"""Data models for goalie statistics."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


def _safe_div(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 3)


@dataclass(frozen=True)
class Game:
    """Minimal representation of a game that goalie stats belong to."""

    identifier: str
    date: Optional[date] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    venue: Optional[str] = None


@dataclass(frozen=True)
class GoalieAppearance:
    """Single goalie appearance for one game."""

    game_id: str
    goalie: str
    team: str
    opponent: str
    saves: int
    shots_against: int
    goals_against: int
    time_on_ice_seconds: Optional[int] = None
    raw_time: Optional[str] = field(default=None, repr=False)

    @property
    def save_percentage(self) -> Optional[float]:
        return _safe_div(self.saves, self.shots_against)


@dataclass(frozen=True)
class GoalieSummary:
    """Aggregated goalie statistics across all appearances."""

    goalie: str
    team: str
    games: int
    saves: int
    shots_against: int
    goals_against: int
    time_on_ice_seconds: int

    @property
    def save_percentage(self) -> Optional[float]:
        return _safe_div(self.saves, self.shots_against)
