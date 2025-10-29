"""Simple floorball goalie stats scraper.

This script downloads the fixture list from statistik.innebandy.se, follows each
match link, extracts goalie appearance statistics, and stores the result in an
Excel workbook.  The implementation purposely avoids advanced dependencies so
it stays easy to run on macOS or any other platform with Python 3.10+
installed.

The output workbook contains one sheet with normalized goalie appearances and a
second sheet summarising per-game metadata.  A small helper function also
creates a save-percentage timeline plot for each goalie using matplotlib if it
is available on the machine.

Usage
-----
python main.py --season-url <fixture url> --output goalie_stats.xlsx

If no arguments are provided the script uses a reasonable default fixture list.
"""
from __future__ import annotations

import argparse
import contextlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# -------------------------
# Configuration defaults
# -------------------------

DEFAULT_FIXTURE_URL = (
    "https://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701"
)
DEFAULT_OUTPUT = Path("goalie_stats.xlsx")
DEFAULT_PLOT = Path("goalie_savepct_timeline.png")

# Patterns that help the parser navigate pages that vary slightly each season.
GOALIE_HEADER = re.compile(r"(Goal(ie|keeper)s?|Målvakter)", re.I)
TEAM_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblHomeTeam",
    "div.gameHeader .homeTeam",
    "div#homeTeam",
    "td.homeTeamName",
]
AWAY_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblAwayTeam",
    "div.gameHeader .awayTeam",
    "div#awayTeam",
    "td.awayTeamName",
]
DATE_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblMatchDate",
    "div.gameHeader .date",
    "div#matchDate",
    "td:contains('Spelades')",
]
GAME_LINK_KEYWORDS = ["scr=game", "scr=result", "matchid=", "fmid=", "gameid="]
GAME_ID_PATTERNS = [
    re.compile(pat, re.I)
    for pat in (
        r"[?&](?:GameID|gameId|gameid)=(?P<id>[A-Za-z0-9\-]+)",
        r"[?&](?:fmid|FMID)=(?P<id>[A-Za-z0-9\-]+)",
        r"[?&](?:matchid|MatchId)=(?P<id>[A-Za-z0-9\-]+)",
        r"[?&](?:gameguid|GameGuid)=(?P<id>[A-Za-z0-9\-]+)",
    )
]


@dataclass
class Game:
    game_id: str
    url: str
    date: Optional[datetime]
    home_team: Optional[str]
    away_team: Optional[str]


@dataclass
class Appearance:
    game_id: str
    goalie_name: str
    team_name: Optional[str]
    saves: Optional[int]
    shots_against: Optional[int]
    goals_against: Optional[int]
    save_pct: Optional[float]
    time_on_ice_seconds: Optional[int]


# -------------------------
# HTTP helpers
# -------------------------

def build_session() -> requests.Session:
    """Return a basic requests session with a sensible user agent."""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
    )
    return session


def fetch_html(session: requests.Session, url: str) -> BeautifulSoup:
    """Fetch *url* and return a parsed BeautifulSoup document."""

    response = session.get(url, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


# -------------------------
# Parsing helpers
# -------------------------

def iter_game_links(doc: BeautifulSoup, base_url: str) -> Iterator[str]:
    """Yield absolute URLs to game pages found in the fixture list."""

    for anchor in doc.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href:
            continue
        if any(keyword in href.lower() for keyword in GAME_LINK_KEYWORDS):
            yield urljoin(base_url, href)


def parse_game_id(url: str) -> str:
    """Return a stable identifier derived from *url*."""

    for pattern in GAME_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group("id")
    parsed = urlparse(url)
    candidate = f"{parsed.path}?{parsed.query}".strip("?")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", candidate).strip("_")
    return cleaned[-80:] or "unknown_game"


def first_text(doc: BeautifulSoup, selectors: Iterable[str]) -> Optional[str]:
    """Return the stripped text of the first matching CSS selector."""

    for selector in selectors:
        element = doc.select_one(selector)
        if element:
            text = element.get_text(strip=True)
            if text:
                return text
    return None


def parse_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%d %B %Y", "%Y-%m-%d %H:%M:%S"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(raw, fmt)
    # dateutil offers better coverage but is an optional dependency, so only
    # import it when we truly need to.
    with contextlib.suppress(ImportError, ValueError):
        from dateutil import parser as date_parser

        return date_parser.parse(raw)
    return None


def time_to_seconds(raw: str | None) -> Optional[int]:
    if not raw:
        return None
    parts = raw.strip().split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None
    return None


def as_int(raw: str | None) -> Optional[int]:
    if raw is None:
        return None
    raw = raw.strip().replace("\u2212", "-")
    if raw in {"", "-", "—"}:
        return None
    with contextlib.suppress(ValueError):
        return int(raw)
    ratio = re.match(r"^(\d+)\s*/\s*(\d+)$", raw)
    if ratio:
        return int(ratio.group(1))
    return None


def compute_save_percentage(saves: Optional[int], shots: Optional[int]) -> Optional[float]:
    if saves is None or shots is None or shots == 0:
        return None
    pct = saves / float(shots)
    return max(0.0, min(1.0, pct))


def extract_goalie_tables(doc: BeautifulSoup) -> List[BeautifulSoup]:
    tables: List[BeautifulSoup] = []
    for heading in doc.find_all(text=GOALIE_HEADER):
        table = heading.find_parent()
        while table and table.name != "table":
            table = table.find_next("table")
        if table and table not in tables:
            tables.append(table)
    for table in doc.select("table.goalkeepers, table#goalkeepers"):
        if table not in tables:
            tables.append(table)
    return tables


def parse_table_headers(table: BeautifulSoup) -> List[str]:
    header_row = table.find("tr")
    if not header_row:
        return []
    headers = [cell.get_text(strip=True).lower() for cell in header_row.find_all(["th", "td"])]
    return headers


def deduce_team_name(table: BeautifulSoup, default: Optional[str]) -> Optional[str]:
    caption = table.find("caption")
    if caption:
        text = caption.get_text(strip=True)
        if text:
            return text
    preceding = table.find_previous(string=True)
    if preceding:
        text = preceding.strip()
        if text:
            return text
    return default


def parse_goalie_rows(
    table: BeautifulSoup, game_id: str, default_team: Optional[str]
) -> Iterator[Appearance]:
    headers = parse_table_headers(table)
    rows = table.find_all("tr")[1:]
    team_name = deduce_team_name(table, default_team)

    def find_index(*candidates: str) -> Optional[int]:
        for candidate in candidates:
            candidate = candidate.lower()
            with contextlib.suppress(ValueError):
                return headers.index(candidate)
        for idx, header in enumerate(headers):
            for candidate in candidates:
                if candidate.lower() in header:
                    return idx
        return None

    name_idx = find_index("spelare", "player", "målvakt", "goalkeeper")
    shots_idx = find_index("skott", "shots", "sa")
    goals_idx = find_index("mål", "ga", "insläppta")
    saves_idx = find_index("räddningar", "saves")
    time_idx = find_index("tid", "toi")

    for row in rows:
        cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
        if not cells or (name_idx is not None and not cells[name_idx]):
            continue
        name = cells[name_idx] if name_idx is not None else cells[0]
        saves = as_int(cells[saves_idx]) if saves_idx is not None else None
        shots = as_int(cells[shots_idx]) if shots_idx is not None else None
        goals = as_int(cells[goals_idx]) if goals_idx is not None else None
        toi = time_to_seconds(cells[time_idx]) if time_idx is not None else None
        yield Appearance(
            game_id=game_id,
            goalie_name=name,
            team_name=team_name,
            saves=saves,
            shots_against=shots,
            goals_against=goals,
            save_pct=compute_save_percentage(saves, shots),
            time_on_ice_seconds=toi,
        )


def parse_game(doc: BeautifulSoup, url: str) -> tuple[Game, List[Appearance]]:
    game_id = parse_game_id(url)
    date_text = first_text(doc, DATE_SELECTORS)
    home = first_text(doc, TEAM_SELECTORS)
    away = first_text(doc, AWAY_SELECTORS)
    game = Game(
        game_id=game_id,
        url=url,
        date=parse_date(date_text),
        home_team=home,
        away_team=away,
    )

    appearances: List[Appearance] = []
    for idx, table in enumerate(extract_goalie_tables(doc)):
        team_hint: Optional[str] = None
        table_text = table.get_text(" ", strip=True)
        if home and home in table_text:
            team_hint = home
        elif away and away in table_text:
            team_hint = away
        elif home or away:
            team_hint = home if idx == 0 else away
        appearances.extend(parse_goalie_rows(table, game_id, team_hint))
    return game, appearances


# -------------------------
# Output helpers
# -------------------------

def appearances_to_frame(appearances: Iterable[Appearance]) -> pd.DataFrame:
    records = [
        {
            "game_id": app.game_id,
            "goalie": app.goalie_name,
            "team": app.team_name,
            "saves": app.saves,
            "shots_against": app.shots_against,
            "goals_against": app.goals_against,
            "save_pct": app.save_pct,
            "time_on_ice_seconds": app.time_on_ice_seconds,
        }
        for app in appearances
    ]
    return pd.DataFrame.from_records(records)


def games_to_frame(games: Iterable[Game]) -> pd.DataFrame:
    records = [
        {
            "game_id": game.game_id,
            "url": game.url,
            "date": game.date,
            "home_team": game.home_team,
            "away_team": game.away_team,
        }
        for game in games
    ]
    return pd.DataFrame.from_records(records)


def write_excel(
    games: Iterable[Game], appearances: Iterable[Appearance], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        games_to_frame(games).to_excel(writer, sheet_name="games", index=False)
        appearances_to_frame(appearances).to_excel(
            writer, sheet_name="appearances", index=False
        )


def create_plot(appearances: Iterable[Appearance], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib not installed; skipping plot generation")
        return

    df = appearances_to_frame(appearances)
    df = df.dropna(subset=["save_pct", "goalie", "game_id"])
    if df.empty:
        print("No save percentage data found; skipping plot generation")
        return
    pivot = df.pivot_table(
        index="game_id",
        columns="goalie",
        values="save_pct",
        aggfunc="mean",
    ).sort_index()

    plt.figure(figsize=(12, 6))
    pivot.plot(marker="o")
    plt.ylabel("Save %")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -------------------------
# CLI entry-point
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season-url",
        default=DEFAULT_FIXTURE_URL,
        help="Fixture list to scrape (defaults to a known season).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Excel file to write (default: goalie_stats.xlsx)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT,
        help="Optional save-percentage plot (PNG).",
    )

    args = parser.parse_args()

    session = build_session()
    fixture_doc = fetch_html(session, args.season_url)
    game_links = sorted(set(iter_game_links(fixture_doc, args.season_url)))

    games: List[Game] = []
    appearances: List[Appearance] = []
    for url in game_links:
        try:
            game_doc = fetch_html(session, url)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            print(f"Failed to fetch {url}: {exc}")
            continue
        game, game_appearances = parse_game(game_doc, url)
        games.append(game)
        appearances.extend(game_appearances)

    if not games:
        raise SystemExit("No games found; verify the fixture URL.")

    write_excel(games, appearances, args.output)
    print(f"Wrote {args.output.resolve()}")

    with contextlib.suppress(Exception):
        create_plot(appearances, args.plot)
        print(f"Created {args.plot.resolve()}")


if __name__ == "__main__":
    main()
