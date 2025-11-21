"""Toolkit for parsing and aggregating floorball goalie statistics."""

from .models import Game, GoalieAppearance, GoalieSummary
from .parser import parse_goalie_table, ParsingError
from .aggregator import summarize_goalies
from .scraper import collect_goalie_stats

__all__ = [
    "Game",
    "GoalieAppearance",
    "GoalieSummary",
    "ParsingError",
    "parse_goalie_table",
    "collect_goalie_stats",
    "summarize_goalies",
]
