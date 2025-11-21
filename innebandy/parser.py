"""HTML parsing utilities for goalie statistics."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import List, Optional

from .models import GoalieAppearance


class ParsingError(ValueError):
    """Raised when an HTML fragment cannot be interpreted as goalie stats."""


_HEADER_ALIASES = {
    "goalie": {"goalie", "keeper", "målvakt"},
    "team": {"team", "lag"},
    "saves": {"saves", "räddningar"},
    "shots": {"shots against", "shots", "sa", "skott"},
    "goals": {"goals against", "goals", "ga", "insläppta"},
    "time": {"toi", "time", "tid", "time on ice"},
}


def _normalise_header(text: str) -> str:
    cleaned = re.sub(r"[^a-z ]", "", text.strip().lower())
    return " ".join(cleaned.split())


def _find_column_indexes(headers: List[str]) -> dict:
    mapping = {}
    for idx, raw in enumerate(headers):
        normalized = _normalise_header(raw)
        for canonical, aliases in _HEADER_ALIASES.items():
            if normalized == canonical or normalized in aliases:
                mapping.setdefault(canonical, idx)
                break
    return mapping


def _parse_int(raw: str, *, field: str, allow_blank: bool = False) -> Optional[int]:
    text = raw.strip()
    if not text:
        if allow_blank:
            return None
        raise ParsingError(f"Missing required numeric value for '{field}'")
    try:
        return int(text)
    except ValueError as exc:
        raise ParsingError(f"Could not parse integer for '{field}' from {raw!r}") from exc


def _parse_time_to_seconds(raw: str) -> Optional[int]:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ParsingError(f"Unrecognised time format: {raw!r}")
    try:
        numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise ParsingError(f"Non-numeric time component in {raw!r}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0
        if seconds >= 60:
            raise ParsingError(f"Invalid time value (seconds must be <60): {raw!r}")
    else:
        hours, minutes, seconds = numbers
        if seconds >= 60 or minutes >= 60:
            raise ParsingError(f"Invalid time value (mm and ss must be <60): {raw!r}")
    return hours * 3600 + minutes * 60 + seconds


class _TableExtractor(HTMLParser):
    """Lightweight HTML table extractor using the standard library."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._active_table: List[List[str]] | None = None
        self._active_row: List[str] | None = None
        self._cell_parts: List[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag == "table":
            self._active_table = []
        elif tag == "tr" and self._active_table is not None:
            self._active_row = []
        elif tag in {"td", "th"} and self._active_row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"td", "th"} and self._cell_parts is not None and self._active_row is not None:
            text = "".join(self._cell_parts).strip()
            self._active_row.append(text)
            self._cell_parts = None
        elif tag == "tr" and self._active_row is not None and self._active_table is not None:
            if any(cell.strip() for cell in self._active_row):
                self._active_table.append(self._active_row)
            self._active_row = None
        elif tag == "table" and self._active_table is not None:
            if self._active_table:
                self.tables.append(self._active_table)
            self._active_table = None


def parse_goalie_table(
    html: str,
    *,
    game_id: str,
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> List[GoalieAppearance]:
    """Parse a goalie statistics table.

    The parser is tolerant to column order and missing optional fields. At minimum
    the table must contain headers for goalie names, saves, and goals against. If
    shots against is omitted it will be derived from saves + goals against.
    """

    extractor = _TableExtractor()
    extractor.feed(html)
    if not extractor.tables:
        raise ParsingError("No table elements found in supplied HTML")

    for rows in extractor.tables:
        if len(rows) < 2:
            continue
        header_row, data_rows = rows[0], rows[1:]
        column_indexes = _find_column_indexes(header_row)
        if "goalie" not in column_indexes or "saves" not in column_indexes:
            continue
        if "goals" not in column_indexes:
            continue

        appearances: List[GoalieAppearance] = []
        for data in data_rows:
            if len(data) < len(header_row):
                continue
            goalie_name = data[column_indexes["goalie"]].strip()
            if not goalie_name:
                continue
            saves = _parse_int(data[column_indexes["saves"]], field="saves")
            goals_cell = data[column_indexes["goals"]]
            goals_against = _parse_int(goals_cell, field="goals against", allow_blank=True)
            shots_against = None
            if "shots" in column_indexes:
                shots_against = _parse_int(
                    data[column_indexes["shots"]], field="shots against", allow_blank=True
                )

            if goals_against is None:
                goals_against = 0 if shots_against is None else max(shots_against - saves, 0)

            derived_shots = saves + goals_against
            if shots_against is None:
                shots_against = derived_shots
            elif shots_against != derived_shots:
                raise ParsingError(
                    f"Shots against mismatch for {goalie_name}: saves + goals != shots"
                )

            team = ""
            if "team" in column_indexes:
                team = data[column_indexes["team"]].strip()
            if not team:
                raise ParsingError(f"Missing team information for {goalie_name}")

            opponent = ""
            if home_team and team.lower() == home_team.lower():
                opponent = away_team or ""
            elif away_team and team.lower() == away_team.lower():
                opponent = home_team or ""

            time_on_ice_seconds = None
            if "time" in column_indexes:
                time_on_ice_seconds = _parse_time_to_seconds(data[column_indexes["time"]])

            appearances.append(
                GoalieAppearance(
                    game_id=game_id,
                    goalie=goalie_name,
                    team=team,
                    opponent=opponent,
                    saves=saves,
                    shots_against=shots_against,
                    goals_against=goals_against,
                    time_on_ice_seconds=time_on_ice_seconds,
                    raw_time=data[column_indexes.get("time", 0)] if "time" in column_indexes else None,
                )
            )
        if appearances:
            return appearances

    raise ParsingError("No goalie statistics table found with required columns")
