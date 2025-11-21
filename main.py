"""Command line interface for parsing goalie stats from HTML files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from innebandy import ParsingError, parse_goalie_table, summarize_goalies


def _load_config(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise SystemExit("Config must be a JSON array")
    return payload


def _load_html(entry: Dict[str, Any], base_dir: Path) -> str:
    if "html" in entry:
        return entry["html"]
    if "html_path" in entry:
        return (base_dir / entry["html_path"]).read_text()
    raise SystemExit("Each config entry needs an 'html' or 'html_path' field")


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


def run(config_path: Path, output_path: Path) -> Dict[str, Any]:
    base_dir = config_path.parent
    config = _load_config(config_path)
    appearances = []
    for entry in config:
        if not isinstance(entry, dict):
            raise SystemExit("Config entries must be objects")
        game_id = entry.get("id")
        home = entry.get("home")
        away = entry.get("away")
        if not game_id:
            raise SystemExit("Each entry requires an 'id' field")
        if not entry.get("team_column", True) and not (home and away):
            raise SystemExit("Home and away must be provided when team column is missing")
        html = _load_html(entry, base_dir)
        try:
            appearances.extend(
                parse_goalie_table(html, game_id=game_id, home_team=home, away_team=away)
            )
        except ParsingError as exc:
            raise SystemExit(f"Failed to parse game {game_id}: {exc}") from exc

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
        "--output", default="stats.json", help="Destination for rendered JSON output"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(Path(args.config), Path(args.output))


if __name__ == "__main__":
    main()
