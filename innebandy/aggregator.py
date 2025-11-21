"""Aggregation helpers for goalie statistics."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Tuple

from .models import GoalieAppearance, GoalieSummary


def summarize_goalies(appearances: Iterable[GoalieAppearance]) -> List[GoalieSummary]:
    bucket: defaultdict[Tuple[str, str], list[GoalieAppearance]] = defaultdict(list)
    for appearance in appearances:
        key = (appearance.goalie, appearance.team)
        bucket[key].append(appearance)

    summaries: List[GoalieSummary] = []
    for (goalie, team), items in sorted(bucket.items(), key=lambda entry: entry[0]):
        saves = sum(item.saves for item in items)
        shots_against = sum(item.shots_against for item in items)
        goals_against = sum(item.goals_against for item in items)
        time_on_ice_seconds = sum(item.time_on_ice_seconds or 0 for item in items)
        summaries.append(
            GoalieSummary(
                goalie=goalie,
                team=team,
                games=len(items),
                saves=saves,
                shots_against=shots_against,
                goals_against=goals_against,
                time_on_ice_seconds=time_on_ice_seconds,
            )
        )
    return summaries
