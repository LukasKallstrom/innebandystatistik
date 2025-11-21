"""Toolkit for parsing and aggregating floorball goalie statistics."""

from .models import Game, GoalieAppearance, GoalieSummary
from .parser import parse_goalie_table, ParsingError
from .aggregator import summarize_goalies

__all__ = [
    "Game",
    "GoalieAppearance",
    "GoalieSummary",
    "ParsingError",
    "parse_goalie_table",
    "summarize_goalies",
]
