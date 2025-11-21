"""Command line interface for downloading and summarising goalie stats."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from innebandy import summarize_goalies
from innebandy.scraper import collect_goalie_stats, default_fetch


def _appearance_to_dict(appearance) -> Dict[str, Any]:
    return {
        "game_id": appearance.game_id,
        "goalie": appearance.goalie,
        "team": appearance.team,
        "opponent": appearance.opponent,
        "saves": appearance.saves,
        "shots_against": appearance.shots_against,
        "goals_against": appearance.goals_against,
        "time_on_ice_seconds": appearance.time_on_ice_seconds,
        "save_percentage": appearance.save_percentage,
        "raw_time": appearance.raw_time,
    }


def _summary_to_dict(summary) -> Dict[str, Any]:
    return {
        "goalie": summary.goalie,
        "team": summary.team,
        "games": summary.games,
        "saves": summary.saves,
        "shots_against": summary.shots_against,
        "goals_against": summary.goals_against,
        "time_on_ice_seconds": summary.time_on_ice_seconds,
        "save_percentage": summary.save_percentage,
    }


def run(
    fixture_url: str,
    output_path: Path,
    *,
    fetcher=default_fetch,
) -> Dict[str, Any]:
    appearances = collect_goalie_stats(fixture_url, fetcher=fetcher)
    appearances.sort(key=lambda a: (a.game_id, a.team.lower(), a.goalie.lower()))
    summaries = summarize_goalies(appearances)
    payload = {
        "appearances": [_appearance_to_dict(item) for item in appearances],
        "summaries": [_summary_to_dict(item) for item in summaries],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON config describing games")
    parser.add_argument(
        "--fixture-url",
        default="http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701",
        help="Fixture list URL to scan for games",
    )
    parser.add_argument(
        "--output", default="stats.json", help="Destination for rendered JSON output"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.fixture_url, Path(args.output))


if __name__ == "__main__":
    main()
