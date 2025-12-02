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
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# -------------------------
# Configuration defaults
# -------------------------

DEFAULT_FIXTURE_URL = (
    "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40693"
)
DEFAULT_OUTPUT = Path("goalie_stats.xlsx")
DEFAULT_PLOT = Path("goalie_savepct_timeline.png")

# Patterns that help the parser navigate pages that vary slightly each season.
GOALIE_HEADER = re.compile(r"(Goal(ie|keeper)s?|Målvakter)", re.I)

# -------------------------
# Match metadata selectors (updated for your markup)
# -------------------------
TEAM_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblHomeTeam",
    "table.clTblMatchStanding tr th:nth-of-type(1)",  # home team in score header
    "div.gameHeader .homeTeam",
    "div#homeTeam",
    "td.homeTeamName",
]
AWAY_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblAwayTeam",
    "table.clTblMatchStanding tr th:nth-of-type(3)",  # away team in score header
    "div.gameHeader .awayTeam",
    "div#awayTeam",
    "td.awayTeamName",
]
DATE_SELECTORS = [
    "span#ctl00_PlaceHolderMain_lblMatchDate",
    "#iMatchInfo tbody tr:has(td:-soup-contains('Tid')) td:nth-of-type(2) span",  # date/time in Matchinformation
    "div.gameHeader .date",
    "div#matchDate",
    "td:-soup-contains('Spelades')",
    #"td:matches((?i)spelades|date)",
]

# Links and ids
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

# --- Heuristics for player tables used as goalie source (Pos.=MV/Målvakt)
PLAYER_TABLE_HEADER_KEYS = (
    "nr", "namn", "name", "spelare", "player",
    "pos", "position", "femma",
    "mål", "ass", "utv", "poäng",
    "skott", "insl", "insl.mål", "räddn", "räddn.(%)",
    "shots", "ga", "save"
)
POS_GOALIE = re.compile(r"^(mv|målvakt|gk|goal(?:ie|keeper))\.?$", re.I)

# Optional header keywords for parsing goalie tables (narrowed to avoid false positives)
GOALIE_NAME_HEADERS = (
    "spelare",
    "målvakt",
    "målvakter",
    "player",
    "goalkeeper",
    "goalie",
    "namn",
    "name",
)
GOALIE_METRIC_HEADERS = (
    "rädd", "save", "skott", "shot", "ga", "insläppta", "tid", "toi"
)

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
    raw = raw.strip().replace("\u2212", "-").replace("\xa0", "").replace(" ", "")
    if raw in {"", "-", "—"}:
        return None
    with contextlib.suppress(ValueError):
        return int(raw)
    ratio = re.match(r"^(\d+)\s*/\s*(\d+)$", raw)
    if ratio:
        return int(ratio.group(1))
    leading = re.match(r"^(\d+)", raw)
    if leading:
        return int(leading.group(1))
    return None


def compute_save_percentage(saves: Optional[int], shots: Optional[int]) -> Optional[float]:
    if saves is None or shots is None or shots == 0:
        return None
    pct = saves / float(shots)
    return max(0.0, min(1.0, pct))


# ---------- STRICT explicit "Målvakter" tables (avoid Pos. tables)

def extract_goalie_tables(doc: BeautifulSoup) -> List[BeautifulSoup]:
    """Find *only* separate goalie tables (not player tables with Pos.)."""
    tables: List[BeautifulSoup] = []
    seen: set[int] = set()

    def table_headers(table: BeautifulSoup) -> List[str]:
        headers: List[str] = []
        header_section = table.find("thead")
        header_rows = header_section.find_all("tr") if header_section else table.find_all("tr", limit=2)
        for row in header_rows:
            for cell in row.find_all(["th", "td"]):
                text = (cell.get_text(strip=True) or cell.get("data-title") or cell.get("title") or "").lower()
                if text:
                    headers.append(text)
            if headers:
                break
        return headers

    def looks_like_goalie_table(headers: List[str]) -> bool:
        if not headers:
            return False
        hdrs = [h.strip().lower() for h in headers]
        # Exclude player tables that have a position column
        if any("pos" in h for h in hdrs):
            return False
        # Require a name column
        has_name = any(h in ("målvakt", "målvakter", "goalkeeper", "goalie", "spelare", "player", "namn", "name") for h in hdrs)
        # And at least one goalie-specific metric
        has_goalie_metrics = any(("rädd" in h) or ("save" in h) or ("insläppta" in h) or re.search(r"\bga\b", h) for h in hdrs)
        return has_name and has_goalie_metrics

    # First: headings that actually say "Målvakter"
    for heading in doc.find_all(string=GOALIE_HEADER):
        table = heading.find_parent()
        while table and table.name != "table":
            table = table.find_next("table")
        if table:
            headers = table_headers(table)
            if looks_like_goalie_table(headers):
                ident = id(table)
                if ident not in seen:
                    tables.append(table); seen.add(ident)

    # Then: generic sweep with the strict heuristic above
    for table in doc.find_all("table"):
        ident = id(table)
        if ident in seen:
            continue
        headers = table_headers(table)
        if looks_like_goalie_table(headers):
            tables.append(table); seen.add(ident)

    return tables


def parse_table_headers(table: BeautifulSoup) -> List[str]:
    """Select the THEAD row that contains real column names.

    Many pages have a first row like 'Laguppställning <Team> | Statistik i matchen',
    followed by the actual column header row. Prefer the row with typical column keys.
    """
    header_section = table.find("thead")
    rows = header_section.find_all("tr") if header_section else table.find_all("tr", limit=2)
    best_headers: List[str] = []
    best_len = 0

    def norm_cells(cells):
        out = []
        for cell in cells:
            txt = (cell.get_text(strip=True) or cell.get("data-title") or cell.get("title") or "")
            out.append(txt.lower())
        return out

    KEY_HINTS = ("namn", "name", "spelare", "player", "pos", "position", "skott", "insl", "insl.mål", "rädd", "räddn", "save", "ga")

    for row in rows:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        headers = norm_cells(cells)

        # If the row looks like column names -> return it
        if any(any(k in h for k in KEY_HINTS) for h in headers) and len(headers) >= 3:
            return headers

        # Otherwise keep the longest non-empty row as fallback
        nonempty = sum(1 for h in headers if h.strip())
        if nonempty > best_len:
            best_headers, best_len = headers, nonempty

    return best_headers


def deduce_team_name(table: BeautifulSoup, default: Optional[str]) -> Optional[str]:
    # 1) caption if present
    caption = table.find("caption")
    if caption:
        txt = caption.get_text(strip=True)
        if txt:
            return txt

    # 2) first THEAD row: "Laguppställning <TEAM>   Statistik i matchen"
    thead = table.find("thead")
    if thead:
        first_row = thead.find("tr")
        if first_row:
            txt = first_row.get_text(" ", strip=True)
            m = re.search(r"Laguppställning\s+(.+?)(?:\s+Statistik i matchen|$)", txt, flags=re.I)
            if m:
                team = m.group(1).strip()
                team = team.replace("\xa0", " ").strip()
                if team:
                    return team

    # 3) nearest previous text node
    preceding = table.find_previous(string=True)
    if preceding:
        txt = preceding.strip()
        if txt:
            return txt

    return default


def parse_goalie_rows(
    table: BeautifulSoup, game_id: str, default_team: Optional[str]
) -> Iterator[Appearance]:
    """Parse rows from a separate goalie table (no Pos.). Also derive saves when absent."""
    headers = parse_table_headers(table)
    body = table.find("tbody")
    rows = body.find_all("tr") if body else table.find_all("tr")[1:]
    team_name = deduce_team_name(table, default_team)

    def find_index(*candidates: str) -> Optional[int]:
        # exact/contains finder (case-insensitive)
        for candidate in candidates:
            c = candidate.lower()
            with contextlib.suppress(ValueError):
                return headers.index(c)
        for idx, header in enumerate(headers):
            for candidate in candidates:
                if candidate.lower() in header:
                    return idx
        return None

    # pick percent column first and exclude it from saves
    save_pct_idx = find_index("rä%", "rädd%", "save%", "sv%", "räddn.(%)", "räddnings%", "räddnings")

    name_idx  = find_index("spelare", "player", "målvakt", "goalkeeper", "namn", "name")
    shots_idx = find_index("skott", "shots", "sa")
    goals_idx = find_index("insläppta", "insl", "insl.mål", "ga", "mål")

    # find 'saves' but EXCLUDE the percent column
    saves_idx: Optional[int] = None
    for idx, header in enumerate(headers):
        h = header.lower()
        if idx == save_pct_idx:
            continue
        if ("rädd" in h or "save" in h) and "%" not in h and "(%)" not in h:
            saves_idx = idx
            break

    for row in rows:
        cells = []
        for cell in row.find_all(["td", "th"]):
            text = cell.get_text(" ", strip=True)
            if not text:
                text = cell.get("data-title") or cell.get("title") or ""
            cells.append(text)

        if not cells:
            continue
        if name_idx is not None and (name_idx >= len(cells) or not cells[name_idx]):
            continue

        name = cells[name_idx] if name_idx is not None else cells[0]

        def to_int(s: Optional[str]) -> Optional[int]:
            if s is None:
                return None
            s = (
                s.strip()
                 .replace("\u2212", "-")
                 .replace("\u202f", "")
                 .replace("\xa0", "")
                 .replace(" ", "")
            )
            m = re.match(r"^-?\d+", s)
            return int(m.group(0)) if m else None

        saves = to_int(cells[saves_idx]) if (saves_idx is not None and saves_idx < len(cells)) else None
        shots = to_int(cells[shots_idx]) if (shots_idx is not None and shots_idx < len(cells)) else None
        goals = to_int(cells[goals_idx]) if (goals_idx is not None and goals_idx < len(cells)) else None
        toi = None
        time_idx = find_index("tid", "toi")
        if time_idx is not None and time_idx < len(cells):
            toi = time_to_seconds(cells[time_idx])

        # derive saves if missing
        if saves is None and shots is not None and goals is not None:
            saves = shots - goals

        pct: Optional[float] = None
        if save_pct_idx is not None and save_pct_idx < len(cells):
            raw_pct = (
                cells[save_pct_idx]
                .replace("%", "")
                .replace("\u202f", "")
                .replace("\xa0", "")
                .replace(",", ".")
                .strip()
            )
            with contextlib.suppress(ValueError):
                val = float(raw_pct)
                pct = val / (100.0 if val > 1.5 else 1.0)

        if pct is None:
            pct = compute_save_percentage(saves, shots)

        yield Appearance(
            game_id=game_id,
            goalie_name=name,
            team_name=team_name,
            saves=saves,
            shots_against=shots,
            goals_against=goals,
            save_pct=pct,
            time_on_ice_seconds=toi,
        )

# ---------- Player tables filtered by Pos.=MV/Målvakt (fallback)

def extract_player_tables(doc: BeautifulSoup) -> List[BeautifulSoup]:
    """Return tables that look like 'Statistik i matchen' (one per team)."""
    tables: List[BeautifulSoup] = []

    def table_headers(table: BeautifulSoup) -> List[str]:
        headers: List[str] = []
        header_section = table.find("thead")
        header_rows = header_section.find_all("tr") if header_section else table.find_all("tr", limit=2)
        for row in header_rows:
            for cell in row.find_all(["th", "td"]):
                text = (cell.get_text(strip=True) or cell.get("data-title") or cell.get("title") or "").lower()
                if text:
                    headers.append(text)
        return headers

    def looks_like_player_table(headers: List[str]) -> bool:
        if not headers:
            return False
        has_name = any("namn" in h or "name" in h or "spelare" in h or "player" in h for h in headers)
        has_pos  = any("pos" in h or "position" in h for h in headers)
        has_stats = any(("skott" in h or "insl" in h or "rädd" in h or "save" in h) for h in headers)
        return has_name and (has_pos or has_stats)

    for table in doc.find_all("table"):
        headers = table_headers(table)
        if looks_like_player_table(headers):
            tables.append(table)
    return tables


def parse_goalies_from_player_table(
    table: BeautifulSoup, game_id: str, default_team: Optional[str]
) -> Iterator[Appearance]:
    headers = parse_table_headers(table)
    body = table.find("tbody")
    rows = body.find_all("tr") if body else table.find_all("tr")[1:]
    team_name = deduce_team_name(table, default_team)

    def find_index(*cands: str) -> Optional[int]:
        for cand in cands:
            c = cand.lower()
            with contextlib.suppress(ValueError):
                return headers.index(c)
        for idx, header in enumerate(headers):
            if any(c.lower() in header for c in cands):
                return idx
        return None

    name_idx = find_index("namn", "name", "spelare", "player")
    pos_idx  = find_index("pos", "position")
    shots_idx = find_index("skott", "shots")
    ga_idx    = find_index("insl", "insl.mål", "mål", "ga")
    svpct_idx = find_index("räddn", "räddn.(%)", "save%", "sv%", "räddnings")

    for row in rows:
        cells = []
        for cell in row.find_all(["td", "th"]):
            text = cell.get_text(" ", strip=True) or cell.get("data-title") or cell.get("title") or ""
            cells.append(text)

        if not cells:
            continue

        # Filter by goalie position
        pos_val = cells[pos_idx].strip() if (pos_idx is not None and pos_idx < len(cells)) else ""
        if not POS_GOALIE.match(pos_val):
            continue

        name = cells[name_idx] if name_idx is not None else cells[0]

        def to_int(s: str | None) -> Optional[int]:
            if not s:
                return None
            s = s.replace("\u2212", "-").replace("\u202f", "").replace("\xa0", "").replace(" ", "")
            s = s.replace(",", ".")
            m = re.match(r"^(-?\d+)", s)
            return int(m.group(1)) if m else None

        shots = to_int(cells[shots_idx]) if shots_idx is not None and shots_idx < len(cells) else None
        ga    = to_int(cells[ga_idx])    if ga_idx    is not None and ga_idx    < len(cells) else None

        pct: Optional[float] = None
        if svpct_idx is not None and svpct_idx < len(cells):
            raw = (
                cells[svpct_idx]
                .replace("%", "")
                .replace("\u202f", "")
                .replace("\xa0", "")
                .replace(",", ".")
                .strip()
            )
            with contextlib.suppress(ValueError):
                val = float(raw)
                pct = val / (100.0 if val > 1.5 else 1.0)

        saves = shots - ga if (shots is not None and ga is not None) else None
        if pct is None:
            pct = compute_save_percentage(saves, shots)

        yield Appearance(
            game_id=game_id,
            goalie_name=name,
            team_name=team_name,
            saves=saves,
            shots_against=shots,
            goals_against=ga,
            save_pct=pct,
            time_on_ice_seconds=None,  # Usually not present in this view
        )


def _extract_date_text(doc: BeautifulSoup) -> Optional[str]:
    """Return the best-effort date/time string for a game page.

    Some match pages (e.g. statistik.innebandy.se) expose the date inside the
    "Matchinformation" table rather than a dedicated span. A few also embed an
    alternative kickoff time inside an HTML comment near the top of the page.
    Prefer the visible table value, then fall back to a generic ISO-like date
    pattern anywhere on the page.
    """

    date_text = first_text(doc, DATE_SELECTORS)
    if date_text:
        return date_text

    # Fallback: inspect the Matchinformation table manually to avoid brittle
    # CSS selectors when soup-sieve support differs across environments.
    match_info = doc.select_one("#iMatchInfo")
    if match_info:
        # look for a row whose first cell mentions time/date
        for row in match_info.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            heading = cells[0].get_text(" ", strip=True).lower()
            if any(key in heading for key in ("tid", "date", "spelades")):
                raw = cells[1].get_text(" ", strip=True)
                if raw:
                    return raw
        # otherwise try a regex inside the table text (avoids picking the
        # commented-out matchtime at the top of the document)
        table_text = match_info.get_text(" ", strip=True)
        match = re.search(r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?", table_text)
        if match:
            return match.group(0)

    # Last resort: grab the first ISO-like date anywhere in the page
    page_text = doc.get_text(" ", strip=True)
    match = re.search(r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?", page_text)
    if match:
        return match.group(0)
    return None


def parse_game(doc: BeautifulSoup, url: str) -> tuple[Game, List[Appearance]]:
    game_id = parse_game_id(url)
    date_text = _extract_date_text(doc)
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

    # 1) Try explicit goalie tables first (strict detection)
    goalie_tables = extract_goalie_tables(doc)
    if goalie_tables:
        for idx, table in enumerate(goalie_tables):
            team_hint: Optional[str] = None
            table_text = table.get_text(" ", strip=True)
            if home and home in table_text:
                team_hint = home
            elif away and away in table_text:
                team_hint = away
            elif home or away:
                team_hint = home if idx == 0 else away
            appearances.extend(parse_goalie_rows(table, game_id, team_hint))

    # 2) Fallback: derive goalies from player tables (Pos.=MV/Målvakt)
    if not appearances:
        player_tables = extract_player_tables(doc)
        for idx, table in enumerate(player_tables):
            team_hint = None
            if home or away:
                team_hint = home if idx == 0 else away
            appearances.extend(parse_goalies_from_player_table(table, game_id, team_hint))

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
        logger.info("matplotlib not installed; skipping plot generation")
        return

    df = appearances_to_frame(appearances)
    df = df.dropna(subset=["save_pct", "goalie", "game_id"])
    if df.empty:
        logger.info("No save percentage data found; skipping plot generation")
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
# Multi-league dashboard helpers
# -------------------------

def _write_debug_csv(dirpath: Path, games_df: pd.DataFrame, apps_df: pd.DataFrame, tag: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    games_df.to_csv(dirpath / f"games_{tag}.csv", index=False)
    apps_df.to_csv(dirpath / f"appearances_{tag}.csv", index=False)


def scrape_data(season_url: str, debug_csv_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape *season_url* and return dataframes for games and goalie appearances."""

    session = build_session()
    logger.info("Fetching fixture list: %s", season_url)
    fixture_doc = fetch_html(session, season_url)
    game_links = sorted(set(iter_game_links(fixture_doc, season_url)))
    logger.info("Discovered %d unique games", len(game_links))

    games: List[Game] = []
    appearances: List[Appearance] = []

    for index, url in enumerate(game_links, start=1):
        logger.debug("Fetching game %d/%d: %s", index, len(game_links), url)
        try:
            game_doc = fetch_html(session, url)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue
        game, game_apps = parse_game(game_doc, url)
        games.append(game)
        appearances.extend(game_apps)
        if not game_apps:
            logger.debug("No goalie appearances parsed for %s", url)

    if not games:
        raise SystemExit("No games detected. Check that the fixture URL is correct.")

    games_df = games_to_frame(games)
    apps_df = appearances_to_frame(appearances)

    if debug_csv_dir is not None:
        _write_debug_csv(debug_csv_dir, games_df, apps_df, tag="raw")

    return games_df, apps_df


def prepare_cumulative_save_percentages(
    games: pd.DataFrame, appearances: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (cumulative_df, dropped_goalies) with cumulative save% per goalie over time."""

    if games.empty:
        raise ValueError("Games dataframe is empty")
    if appearances.empty:
        raise ValueError("No goalie appearances found (0 rows).")

    required_columns = {"game_id", "goalie"}
    missing_required = required_columns.difference(appearances.columns)
    if missing_required:
        raise ValueError(f"Missing required appearance columns: {', '.join(sorted(missing_required))}")

    for col in ("saves", "shots_against", "goals_against", "save_pct"):
        if col not in appearances.columns:
            appearances[col] = pd.NA

    games = games.copy()
    games["__order"] = range(len(games))

    merged = appearances.merge(
        games[["game_id", "date", "__order"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    for col in ("saves", "shots_against", "goals_against"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["save_pct"] = (
        merged["save_pct"]
        .astype(str)
        .str.strip()
        .str.replace("\u202f", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    merged["save_pct"] = pd.to_numeric(merged["save_pct"], errors="coerce")
    merged.loc[merged["save_pct"] > 1.5, "save_pct"] = merged.loc[merged["save_pct"] > 1.5, "save_pct"] / 100.0

    m = merged

    mask = m["shots_against"].isna() & m["saves"].notna() & m["goals_against"].notna()
    m.loc[mask, "shots_against"] = m.loc[mask, "saves"] + m.loc[mask, "goals_against"]

    mask = m["saves"].isna() & m["shots_against"].notna() & m["goals_against"].notna()
    m.loc[mask, "saves"] = m.loc[mask, "shots_against"] - m.loc[mask, "goals_against"]

    mask = m["saves"].isna() & m["shots_against"].notna() & m["save_pct"].notna()
    m.loc[mask, "saves"] = (m.loc[mask, "save_pct"] * m.loc[mask, "shots_against"]).round().astype("Int64")

    mask = m["shots_against"].isna() & m["goals_against"].notna() & m["save_pct"].notna()
    denom = 1.0 - m.loc[mask, "save_pct"]
    feasible = denom > 1e-9
    idxs = denom[feasible].index
    est_shots = (m.loc[idxs, "goals_against"] / denom.loc[idxs]).round()
    m.loc[idxs, "shots_against"] = est_shots
    m.loc[idxs, "saves"] = (m.loc[idxs, "shots_against"] - m.loc[idxs, "goals_against"]).astype("Int64")

    mask = m["goals_against"].isna() & m["shots_against"].notna() & m["saves"].notna()
    m.loc[mask, "goals_against"] = (m.loc[mask, "shots_against"] - m.loc[mask, "saves"]).astype("Int64")

    missing_dates = m["date"].isna()
    if missing_dates.any():
        base = pd.Timestamp("2000-01-01")
        m["__order"] = pd.to_numeric(m["__order"], errors="coerce")
        m["__order"] = m["__order"].fillna(m["__order"].min(skipna=True)).fillna(0).astype(int)
        m.loc[missing_dates, "date"] = base + pd.to_timedelta(m.loc[missing_dates, "__order"], unit="D")
        logger.warning(
            "Dates missing for %d appearances — using synthetic dates based on match order.",
            int(missing_dates.sum()),
        )

    m = m.dropna(subset=["date", "goalie"])
    m = m[~(m["saves"].isna() & m["shots_against"].isna())]

    mask = m["saves"].isna() & m["shots_against"].notna() & m["goals_against"].notna()
    m.loc[mask, "saves"] = m.loc[mask, "shots_against"] - m.loc[mask, "goals_against"]
    mask = m["shots_against"].isna() & m["saves"].notna() & m["goals_against"].notna()
    m.loc[mask, "shots_against"] = m.loc[mask, "saves"] + m.loc[mask, "goals_against"]

    m = m.dropna(subset=["saves", "shots_against"])
    if m.empty:
        raise ValueError("Insufficient goalie data with saves and shots even after reconstruction and synthetic dates.")

    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "game_id"])

    m["saves"] = pd.to_numeric(m["saves"], errors="coerce").fillna(0).astype(int)
    m["shots_against"] = pd.to_numeric(m["shots_against"], errors="coerce").fillna(0).astype(int)
    m["cumulative_saves"] = m.groupby("goalie")["saves"].cumsum()
    m["cumulative_shots"] = m.groupby("goalie")["shots_against"].cumsum()

    m["cumulative_save_pct"] = pd.NA
    pos_mask = m["cumulative_shots"] > 0
    m.loc[pos_mask, "cumulative_save_pct"] = m.loc[pos_mask, "cumulative_saves"] / m.loc[pos_mask, "cumulative_shots"]

    shot_sum_by_goalie = m.groupby("goalie")["shots_against"].transform("sum")
    has_any_shots = shot_sum_by_goalie > 0
    dropped_goalies = (
        m.loc[~has_any_shots, "goalie"].drop_duplicates().sort_values().tolist()
    )
    m = m[has_any_shots]

    if m.empty:
        raise ValueError(
            "All goalies in the dataset have zero shots against; nothing to plot."
        )

    if "team" in m:
        m["team"] = m["team"].fillna("Unknown")

    return m, dropped_goalies


def summarise_goalies(cumulative: pd.DataFrame) -> pd.DataFrame:
    grouped = cumulative.groupby(["team", "goalie"], dropna=False)

    def _last_valid(series: pd.Series) -> float | None:
        valid = series.dropna()
        if valid.empty:
            return None
        return float(valid.iloc[-1])

    summary = grouped.agg(
        games_played=("game_id", pd.Series.nunique),
        total_shots=("shots_against", "sum"),
        total_saves=("saves", "sum"),
        goals_against=("goals_against", "sum"),
        final_save_pct=("cumulative_save_pct", _last_valid),
    ).reset_index()

    summary["games_played"] = summary["games_played"].astype(int)
    for col in ("total_shots", "total_saves", "goals_against"):
        summary[col] = summary[col].fillna(0).astype(int)

    summary["avg_shots_per_game"] = summary.apply(
        lambda row: (row["total_shots"] / row["games_played"]) if row["games_played"] else 0.0,
        axis=1,
    )

    summary["save_pct_display"] = summary["final_save_pct"].apply(
        lambda value: f"{value:.1%}" if value is not None else "–"
    )

    return summary.sort_values(["team", "goalie"]).reset_index(drop=True)


@dataclass
class LeagueSnapshot:
    key: str
    name: str
    figure: Dict
    summary_records: List[dict]
    goalie_to_team: Dict[str, str]
    trace_goalies: List[str]
    team_options: List[str]
    dropped_goalies: List[str]
    stats: Dict[str, object]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "league"


def _date_range_text(cumulative: pd.DataFrame) -> str:
    date_min = pd.to_datetime(cumulative["date"]).min()
    date_max = pd.to_datetime(cumulative["date"]).max()
    if pd.notna(date_min) and pd.notna(date_max):
        return f"{date_min.date().isoformat()} → {date_max.date().isoformat()}"
    return "–"


def build_figure(cumulative: pd.DataFrame, dropped_goalies: List[str]):
    import importlib.util

    if importlib.util.find_spec("plotly.express") is None:
        raise SystemExit(
            "plotly is required for the interactive visualisation. Install it via 'pip install plotly'."
        )

    import plotly.express as px

    fig = px.line(
        cumulative,
        x="date",
        y="cumulative_save_pct",
        color="goalie",
        line_group="goalie",
        markers=True,
        hover_data={
            "team": True,
            "game_id": True,
            "cumulative_saves": True,
            "cumulative_shots": True,
            "cumulative_save_pct": ":.1%",
        },
        labels={
            "date": "Date",
            "cumulative_save_pct": "Cumulative Save %",
            "goalie": "Goalie",
        },
    )

    fig.update_layout(
        title="Goalie cumulative save percentage over time",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(tickformat=".0%", rangemode="tozero"),
        legend_title_text="Goalie",
    )

    if dropped_goalies:
        text_lines = ["<b>Filtered (0 shots):</b>"] + [f"• {name}" for name in dropped_goalies]
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            align="right",
            text="<br>".join(text_lines),
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=6,
        )

    return fig


def _build_league_snapshot(
    name: str,
    figure,
    cumulative: pd.DataFrame,
    summary: pd.DataFrame,
    dropped_goalies: List[str],
) -> LeagueSnapshot:
    import plotly.io as pio

    cumulative = cumulative.copy()
    cumulative["team"] = cumulative.get("team", "Unknown").fillna("Unknown")

    goalie_team_map = (
        cumulative[["goalie", "team"]]
        .drop_duplicates()
        .sort_values("goalie")
        .assign(team=lambda df: df["team"].fillna("Unknown"))
    )
    goalie_to_team = {row.goalie: row.team for row in goalie_team_map.itertuples(index=False)}

    team_options = sorted({team for team in cumulative["team"].dropna().unique()})
    summary_records = summary.replace({pd.NA: None}).to_dict(orient="records")

    stats = {
        "total_goalies": len({row["goalie"] for row in summary_records}),
        "total_teams": len(team_options),
        "date_range": _date_range_text(cumulative),
        "zero_shot_goalies": len(dropped_goalies),
    }

    figure_json = json.loads(pio.to_json(figure, engine="json"))

    return LeagueSnapshot(
        key=_slugify(name),
        name=name,
        figure=figure_json,
        summary_records=summary_records,
        goalie_to_team=goalie_to_team,
        trace_goalies=[trace.name for trace in figure.data],
        team_options=team_options,
        dropped_goalies=dropped_goalies,
        stats=stats,
    )


def build_dashboard_html(leagues: List[LeagueSnapshot]) -> str:
    """Compose the interactive dashboard HTML shell."""

    if not leagues:
        raise ValueError("At least one league is required to build the dashboard")

    first = leagues[0]
    dropdown_options = "".join(
        f'<option value="{team}">{team}</option>' for team in first.team_options
    )

    league_json = json.dumps([league.__dict__ for league in leagues], ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Goalie Save Percentage Dashboard</title>
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
    <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\" />
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #0f172a;
        --bg-panel: rgba(15, 23, 42, 0.75);
        --bg-light: #f8fafc;
        --border: rgba(148, 163, 184, 0.35);
        --text: #0f172a;
        --text-muted: #475569;
        --accent: #2563eb;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}

      body {{
        margin: 0;
        padding: 0;
        background: linear-gradient(160deg, #020617, #0f172a 55%, #1e293b);
        color: white;
        min-height: 100vh;
      }}

      .page {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 2.5rem 1.5rem 4rem;
        display: flex;
        flex-direction: column;
        gap: 2.5rem;
      }}

      header {{
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }}

      h1 {{
        margin: 0;
        font-size: clamp(1.8rem, 4vw, 2.4rem);
        letter-spacing: -0.02em;
      }}

      p.description {{
        margin: 0;
        color: rgba(255, 255, 255, 0.75);
        max-width: 820px;
        line-height: 1.6;
      }}

      .panel {{
        background: var(--bg-panel);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(10px);
      }}

      .controls {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        align-items: flex-end;
      }}

      label {{
        display: block;
        margin-bottom: 0.35rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
        letter-spacing: 0.01em;
      }}

      select, input {{
        width: 100%;
        padding: 0.65rem 0.85rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.06);
        color: white;
        font-size: 0.95rem;
        transition: border-color 0.2s ease, background 0.2s ease;
      }}

      select:focus, input:focus {{
        outline: none;
        border-color: rgba(37, 99, 235, 0.8);
        background: rgba(255, 255, 255, 0.1);
      }}

      button {{
        background: linear-gradient(120deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.01em;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.2s ease;
        box-shadow: 0 12px 35px rgba(37, 99, 235, 0.25);
      }}

      button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.32);
      }}

      button:active {{
        transform: translateY(0);
        opacity: 0.9;
      }}

      #savepct-chart {{
        width: 100%;
        height: 580px;
      }}

      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
      }}

      .stat-card {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.9rem 1rem;
      }}

      .stat-label {{
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
      }}

      .stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
      }}

      th, td {{
        padding: 0.75rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      }}

      th {{
        color: rgba(255, 255, 255, 0.8);
        font-weight: 700;
        cursor: pointer;
        white-space: nowrap;
      }}

      td {{
        color: rgba(255, 255, 255, 0.85);
      }}

      tr:hover td {{
        background: rgba(255, 255, 255, 0.03);
      }}

      .tag {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
      }}

      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.2);
        color: white;
        border: 1px solid rgba(37, 99, 235, 0.45);
        font-size: 0.85rem;
      }}

      .muted {{
        color: rgba(255, 255, 255, 0.65);
      }}

      .dropped-list {{
        margin-top: 0.4rem;
        color: rgba(255, 255, 255, 0.75);
        line-height: 1.6;
      }}

      footer {{
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.9rem;
        text-align: center;
        padding-bottom: 2rem;
      }}
    </style>
  </head>
  <body>
    <div class=\"page\">
      <header>
        <h1>Goalie Save Percentage Dashboard</h1>
        <p class=\"description\">Interactive view of goalie cumulative save percentage across leagues. Use the dropdowns below to switch leagues, filter by team, and search goalies.</p>
      </header>

      <div class=\"panel\">
        <div class=\"controls\">
          <div>
            <label for=\"league-selector\">League</label>
            <select id=\"league-selector\"></select>
          </div>
          <div>
            <label for=\"team-filter\">Team filter</label>
            <select id=\"team-filter\">
              <option value=\"ALL\">All teams</option>
              {dropdown_options}
            </select>
          </div>
          <div>
            <label for=\"goalie-search\">Search goalie</label>
            <input id=\"goalie-search\" type=\"search\" placeholder=\"Type a goalie name...\" />
          </div>
          <div>
            <label>&nbsp;</label>
            <button id=\"reset-view\">Reset view</button>
          </div>
        </div>

        <div id=\"savepct-chart\"></div>

        <div class=\"stats-grid\" id=\"league-stats\"></div>
      </div>

      <div class=\"panel\">
        <div style=\"display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;\">
          <h2 style=\"margin: 0;\">Goalie summary</h2>
          <div class=\"tag\" id=\"active-league-name\"></div>
        </div>
        <table>
          <thead>
            <tr>
              <th data-key=\"goalie\" class=\"sortable\">Goalie</th>
              <th data-key=\"team\" class=\"sortable\">Team</th>
              <th data-key=\"games_played\" class=\"sortable\">Games</th>
              <th data-key=\"total_shots\" class=\"sortable\">Shots</th>
              <th data-key=\"goals_against\" class=\"sortable\">GA</th>
              <th data-key=\"avg_shots_per_game\" class=\"sortable\">Avg shots</th>
              <th data-key=\"final_save_pct\" class=\"sortable\">Save %</th>
            </tr>
          </thead>
          <tbody id=\"summary-body\"></tbody>
        </table>
        <div class=\"dropped-list\" id=\"dropped-goalies\"></div>
      </div>

      <footer>
        Built with <span aria-hidden=\"true\">🏑</span> using statistik.innebandy.se data. Cumulative save percentage is calculated from saves/shots over time; goalies with zero shots are excluded from the chart but listed above.
      </footer>
    </div>

    <script>
      const leagues = {league_json};

      function populateLeagueSelector() {{
        const selector = document.getElementById('league-selector');
        selector.innerHTML = '';
        leagues.forEach((league) => {{
          const option = document.createElement('option');
          option.value = league.key;
          option.textContent = league.name;
          selector.appendChild(option);
        }});
      }}

      function formatNumber(n) {{
        return new Intl.NumberFormat('sv-SE').format(n);
      }}

      function renderStats(league) {{
        const stats = league.stats;
        const container = document.getElementById('league-stats');
        container.innerHTML = '';
        const entries = [
          ['Goalies', stats.total_goalies],
          ['Teams', stats.total_teams],
          ['Date range', stats.date_range],
          ['Zero-shot goalies', stats.zero_shot_goalies],
        ];
        entries.forEach(([label, value]) => {{
          const card = document.createElement('div');
          card.className = 'stat-card';
          card.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>`;
          container.appendChild(card);
        }});
      }}

      function renderTable(records) {{
        const tbody = document.getElementById('summary-body');
        tbody.innerHTML = '';
        records.forEach((row) => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${row.goalie}</td>
            <td>${row.team ?? ''}</td>
            <td>${row.games_played}</td>
            <td>${formatNumber(row.total_shots)}</td>
            <td>${formatNumber(row.goals_against)}</td>
            <td>${row.avg_shots_per_game.toFixed(1)}</td>
            <td>${row.save_pct_display}</td>
          `;
          tbody.appendChild(tr);
        }});
      }}

      function renderDropped(list) {{
        const target = document.getElementById('dropped-goalies');
        if (!list || list.length === 0) {{
          target.textContent = '';
          return;
        }}
        target.innerHTML = `<div class="muted">Filtered out (0 shots): ${list.join(', ')}</div>`;
      }}

      function sortBy(key, ascending) {{
        return (a, b) => {{
          const av = a[key];
          const bv = b[key];
          if (av === bv) return 0;
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          const comp = av > bv ? 1 : -1;
          return ascending ? comp : -comp;
        }};
      }}

      function applyFilters(records) {{
        const team = document.getElementById('team-filter').value;
        const query = document.getElementById('goalie-search').value.trim().toLowerCase();
        return records.filter((row) => {{
          const matchesTeam = team === 'ALL' || row.team === team;
          const matchesQuery = !query || row.goalie.toLowerCase().includes(query);
          return matchesTeam && matchesQuery;
        }});
      }}

      let summaryData = [];
      let sortState = {{ key: 'team', ascending: true }};
      let traceGoalies = [];

      function restyleChart() {{
        if (traceGoalies.length === 0) return;
        const team = document.getElementById('team-filter').value;
        const query = document.getElementById('goalie-search').value.trim().toLowerCase();
        const visibility = traceGoalies.map((goalie) => {{
          const matchesTeam = team === 'ALL' || (goalie in currentLeague.goalie_to_team && currentLeague.goalie_to_team[goalie] === team);
          const matchesQuery = !query || goalie.toLowerCase().includes(query);
          return matchesTeam && matchesQuery;
        }});
        Plotly.restyle('savepct-chart', {{ visible: visibility }}, Array.from(traceGoalies.keys()));
      }}

      function loadLeague(key) {{
        currentLeague = leagues.find((league) => league.key === key) ?? leagues[0];
        summaryData = currentLeague.summary_records;
        traceGoalies = currentLeague.trace_goalies;
        document.getElementById('active-league-name').textContent = currentLeague.name;

        const teamSelect = document.getElementById('team-filter');
        teamSelect.innerHTML = '<option value="ALL">All teams</option>' + currentLeague.team_options.map((team) => `<option value="${team}">${team}</option>`).join('');

        const sorted = summaryData.sort(sortBy(sortState.key, sortState.ascending));
        renderTable(applyFilters(sorted));
        renderDropped(currentLeague.dropped_goalies);
        renderStats(currentLeague);

        Plotly.newPlot('savepct-chart', currentLeague.figure.data, currentLeague.figure.layout, {{ responsive: true }});
        restyleChart();
      }}

      let currentLeague = null;

      window.addEventListener('DOMContentLoaded', () => {{
        populateLeagueSelector();
        loadLeague(leagues[0].key);

        document.getElementById('league-selector').addEventListener('change', (event) => {{
          loadLeague(event.target.value);
        }});

        document.getElementById('team-filter').addEventListener('change', () => {{
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        }});

        document.getElementById('goalie-search').addEventListener('input', () => {{
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        }});

        document.getElementById('reset-view').addEventListener('click', () => {{
          sortState = {{ key: 'team', ascending: true }};
          document.getElementById('team-filter').value = 'ALL';
          document.getElementById('goalie-search').value = '';
          renderTable(summaryData.sort(sortBy(sortState.key, sortState.ascending)));
          traceGoalies.forEach((goalie, index) => {{
            Plotly.restyle('savepct-chart', {{ visible: true }}, [index]);
          }});
        }});

        document.querySelectorAll('th.sortable').forEach((th) => {{
          th.addEventListener('click', () => {{
            const key = th.dataset.key;
            if (sortState.key === key) {{
              sortState.ascending = !sortState.ascending;
            }} else {{
              sortState = {{ key, ascending: true }};
            }}
            const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
            renderTable(filtered);
          }});
        }});

        Plotly.newPlot('savepct-chart', leagues[0].figure.data, leagues[0].figure.layout, {{ responsive: true }});
        Plotly.d3.select('#savepct-chart').on('plotly_doubleclick', () => {{
          document.getElementById('reset-view').click();
        }});
      }});
    </script>
  </body>
</html>"""


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
    parser.add_argument(
        "--league",
        action="append",
        metavar="NAME=FIXTURE_URL",
        help="Add a league by providing a friendly name and fixture URL. Repeatable.",
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        help="Generate an interactive HTML dashboard at this path (default: index.html).",
    )
    parser.add_argument(
        "--debug-csv",
        type=Path,
        default=None,
        help="Directory to write raw CSV exports when building the dashboard.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for detailed progress information.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output, showing only warnings and errors.",
    )

    args = parser.parse_args()

    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.league and args.dashboard is None:
        args.dashboard = Path("index.html")

    if args.league or args.dashboard:
        league_specs = args.league or [f"Primary league={args.season_url}"]
        dashboard_path = args.dashboard or Path("index.html")

        league_snapshots: List[LeagueSnapshot] = []
        for spec in league_specs:
            if "=" not in spec:
                parser.error("--league values must be in the format 'Name=FIXTURE_URL'")
            name, url = spec.split("=", 1)
            name = name.strip() or "League"
            url = url.strip()
            games_df, appearances_df = scrape_data(url, debug_csv_dir=args.debug_csv)
            cumulative, dropped_goalies = prepare_cumulative_save_percentages(games_df, appearances_df)
            figure = build_figure(cumulative, dropped_goalies)
            summary = summarise_goalies(cumulative)
            league_snapshots.append(
                _build_league_snapshot(name, figure, cumulative, summary, dropped_goalies)
            )

        html = build_dashboard_html(league_snapshots)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(html, encoding="utf-8")
        logger.info("Interactive dashboard written to %s", dashboard_path.resolve())
        return

    session = build_session()
    logger.info("Fetching fixture list: %s", args.season_url)
    fixture_doc = fetch_html(session, args.season_url)
    game_links = sorted(set(iter_game_links(fixture_doc, args.season_url)))
    logger.info("Found %d unique game links", len(game_links))

    games: List[Game] = []
    appearances: List[Appearance] = []
    total_games = len(game_links)
    for idx, url in enumerate(game_links, start=1):
        logger.info("Fetching game %d/%d: %s", idx, total_games, url)
        try:
            game_doc = fetch_html(session, url)
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue
        game, game_appearances = parse_game(game_doc, url)
        games.append(game)
        appearances.extend(game_appearances)
        logger.debug(
            "Parsed %d goalie appearances from %s",
            len(game_appearances),
            url,
        )
        if not game_appearances:
            logger.warning("No goalie statistics detected for %s", url)

    if not games:
        raise SystemExit("No games found; verify the fixture URL.")

    write_excel(games, appearances, args.output)
    logger.info("Wrote %s", args.output.resolve())

    with contextlib.suppress(Exception):
        create_plot(appearances, args.plot)
        logger.info("Created %s", args.plot.resolve())


if __name__ == "__main__":
    main()
