#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Floorball Goalie Save-Percentage Scraper & Grapher

What this program does:
1) Scrapes the fixture list at statistik.innebandy.se for all game links.
2) Visits each game page and extracts goalie appearance stats (per goalie, per team),
   including cases where a team swaps goalies mid-game.
3) Stores normalized data into a SQLite database:
   - games (one row per game)
   - goalies (unique goalie identities)
   - appearances (one row per goalie appearance per game)
4) Builds a timeline graph of save percentage by goalie over the season.

Key design notes:
- Robust against transient network errors (retries, backoff, timeouts).
- Parallelized fetching for speed (ThreadPoolExecutor with a guarded session).
- Optional HTTP caching (requests-cache) to accelerate repeated runs.
- Extensive comments to explain the logic.
- Defensive parsing: multiple CSS/XPath selectors tried in priority order.
- Edge-case handling: division by zero, missing stats, partial data, swapped goalies.

Author: (you)
"""

from __future__ import annotations

import re
import sys
import time
import math
import json
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import requests_cache  # Optional but recommended
except Exception:
    requests_cache = None  # Graceful fallback if not installed

from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from dateutil import parser as dateparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------------------
# Configuration (edit if needed)
# ----------------------------

FIXTURE_URL = "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701"

# Maximum concurrent game fetches – tune up/down depending on your connection & site tolerance
MAX_WORKERS = 10

# HTTP timeouts
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20

# If the figure would contain too many lines, you *can* cap it for readability.
# To strictly include all goalies, set MAX_GOALIES_IN_GRAPH = None
MAX_GOALIES_IN_GRAPH: Optional[int] = None  # e.g., 40

# Output figure path
OUTPUT_FIG = "goalie_savepct_timeline.png"

# SQLite DB file
DB_PATH = "floorball_goalies.sqlite3"

# Log level: DEBUG for more detail, INFO for normal, WARNING to reduce chatter
LOG_LEVEL = logging.INFO

# Known selectors/patterns used to locate elements on pages.
# If the site changes, tweak these first (kept together for maintainability).
KNOWN_SELECTORS = {
    # Fixture list page: find all links to individual game pages
    # We look for anchors likely pointing to game result/details:
    "game_link_hrefs_contains": [
        "scr=game",        # generic guess
        "scr=result",      # typical result page
        "GameID=",         # some backends use GameID
        "fmid=",           # innebandy often uses fmid
        "matchid=",        # alternate
        "gameguid=",       # alternate
    ],
    # On the game page: team & date meta
    "game_date_selectors": [
        "div.gameHeader .date",          # hypothetical
        "div#gameheader .date",
        "span#ctl00_PlaceHolderMain_lblMatchDate",  # ASP.NET style
        "div#matchDate",
        "td:contains('Spelades')",       # Swedish pages sometimes have this text
    ],
    "home_team_selectors": [
        "div.gameHeader .homeTeam",
        "span#ctl00_PlaceHolderMain_lblHomeTeam",
        "div#homeTeam",
        "td.homeTeamName",
    ],
    "away_team_selectors": [
        "div.gameHeader .awayTeam",
        "span#ctl00_PlaceHolderMain_lblAwayTeam",
        "div#awayTeam",
        "td.awayTeamName",
    ],
    # Goalie tables usually labeled with words like "Goalkeepers" / "Målvakter".
    # We use headings + following tables or direct table IDs.
    "goalie_section_head_regex": r"(Goal(keeper|ie)s?|Målvakter|M\u00E5lvakter)",
    "goalie_table_selectors": [
        # Known common patterns (IDs/classes often ASP.NET generated, so we search widely)
        "table.goalkeepers",
        "table#goalkeepers",
        "table:contains('Målvakter')",
        "table:contains('Goalkeepers')",
    ],
}

# Fallback patterns for extracting a "game id" from a URL reliably enough for a unique key.
KNOWN_GAME_PATTERNS = [
    r"[?&](?:GameID|gameId|gameid)=(?P<id>[A-Za-z0-9\-]+)",
    r"[?&](?:fmid|FMID)=(?P<id>[A-Za-z0-9\-]+)",
    r"[?&](?:matchid|MatchId)=(?P<id>[A-Za-z0-9\-]+)",
    r"[?&](?:gameguid|GameGuid)=(?P<id>[A-Za-z0-9\-]+)",
]


# ----------------------------
# Data models
# ----------------------------

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
    team_name: str
    # Basic stat lines
    saves: Optional[int]
    shots_against: Optional[int]
    goals_against: Optional[int]
    save_pct: Optional[float]   # 0..1
    toi_seconds: Optional[int]  # time on ice in seconds; None if unknown


# ----------------------------
# Logging setup
# ----------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ----------------------------
# HTTP session with retries & (optional) cache
# ----------------------------

def build_session() -> requests.Session:
    """
    Build a hardened requests Session with:
      - optional caching (requests-cache) to accelerate repeated runs,
      - retry-with-backoff for transient errors,
      - a reasonable timeout strategy.
    """
    if requests_cache is not None:
        # Cache for 12 hours by default (tweak as needed)
        requests_cache.install_cache("floorball_cache", expire_after=12 * 3600)

    session = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # A sensible desktop UA reduces the chance of anti-bot blocks
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36"
    })
    return session


# ----------------------------
# Utilities
# ----------------------------

def fetch_html(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    """
    Fetch a URL and return a BeautifulSoup document or None on hard failure.
    """
    try:
        r = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except requests.RequestException as e:
        logger.warning("Fetch failed for %s: %s", url, e)
        return None


def parse_game_id_from_url(url: str) -> str:
    """
    Heuristically extract a unique game id from a game URL, using known patterns.
    If nothing found, fall back to a normalized version of the path+query.
    """
    for pat in KNOWN_GAME_PATTERNS:
        m = re.search(pat, url, flags=re.I)
        if m:
            return m.group("id")
    # Fallback: sanitize the last path + query
    parsed = urlparse(url)
    raw = (parsed.path or "") + "?" + (parsed.query or "")
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return raw[-100:]  # keep final portion


def text_or_none(el) -> Optional[str]:
    """
    Extract text from a BeautifulSoup element safely.
    """
    if not el:
        return None
    txt = el.get_text(strip=True)
    return txt or None


def time_str_to_seconds(s: Optional[str]) -> Optional[int]:
    """
    Convert a time string like "59:43" to seconds. Returns None if unknown or malformed.
    """
    if not s:
        return None
    s = s.strip()
    # Handle formats like "59:43" or "1:02:33"
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            return m * 60 + sec
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + sec
    except ValueError:
        return None
    return None


def safe_int(s: Optional[str]) -> Optional[int]:
    """
    Convert text to int if possible (handles '—', '-', '' as None).
    """
    if s is None:
        return None
    s = s.strip().replace("\u2212", "-")  # minus symbol fix if present
    if s in {"", "-", "—"}:
        return None
    try:
        return int(s)
    except ValueError:
        # Sometimes we get "12/15" like "saves/shots"
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
        if m:
            return int(m.group(1))  # caller must handle second part separately
        return None


def compute_save_pct(saves: Optional[int], shots_against: Optional[int]) -> Optional[float]:
    """
    Compute save percentage (0..1) safely. Returns None if not computable.
    """
    if saves is None and shots_against is None:
        return None
    if shots_against is None:
        # If shots unknown but goals known, we can't derive.
        return None
    if shots_against == 0:
        # Define 100% if zero shots faced and no goals against; else 0% if goals recorded.
        if saves is None:
            return 1.0  # Assume perfect if no shots (common in very short TOI) – adjustable decision.
        return 1.0 if saves == 0 else (saves / 1.0)  # fallback
    if saves is None:
        return None
    return max(0.0, min(1.0, saves / float(shots_against)))


# ----------------------------
# Parsing functions (fixture & game pages)
# ----------------------------

def find_game_links(doc: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract candidate game links from a fixture list page.
    Uses a tolerant approach: collects anchors whose href contains any of a known set of substrings.
    """
    links = []
    for a in doc.find_all("a", href=True):
        href = a["href"]
        low = href.lower()
        if any(key in low for key in KNOWN_SELECTORS["game_link_hrefs_contains"]):
            full = urljoin(base_url, href)
            links.append(full)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in links:
        if url not in seen:
            unique.append(url)
            seen.add(url)
    return unique


def extract_text_via_selectors(doc: BeautifulSoup, selectors: List[str]) -> Optional[str]:
    """
    Try a set of CSS selectors (and pseudo-selectors) to pull out text.
    Recognizes ':contains("text")' by scanning for tags containing a substring.
    """
    for sel in selectors:
        if ":contains(" in sel:
            # Handle custom contains selector: tag:contains('literal')
            m = re.match(r"^\s*([a-zA-Z0-9\#\.\-:_]+)\s*:\s*contains\(\s*'([^']+)'\s*\)\s*$", sel)
            if m:
                tag_query, literal = m.group(1), m.group(2)
                for el in doc.select(tag_query):
                    if el and literal in el.get_text(" ", strip=True):
                        txt = el.get_text(" ", strip=True)
                        if txt:
                            return txt
            continue
        elements = doc.select(sel)
        if elements:
            txt = text_or_none(elements[0])
            if txt:
                return txt
    return None


def parse_game_meta(doc: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[datetime]]:
    """
    Try to parse home team, away team, and date from a game page.
    Returns (home_team, away_team, date_dt)
    """
    home = extract_text_via_selectors(doc, KNOWN_SELECTORS["home_team_selectors"])
    away = extract_text_via_selectors(doc, KNOWN_SELECTORS["away_team_selectors"])
    raw_date = extract_text_via_selectors(doc, KNOWN_SELECTORS["game_date_selectors"])

    dt = None
    if raw_date:
        # The date may include time and words; use dateutil's robust parser
        try:
            dt = dateparser.parse(raw_date, dayfirst=True, fuzzy=True)
        except Exception:
            dt = None
    return home, away, dt


def find_goalie_tables(doc: BeautifulSoup) -> List[BeautifulSoup]:
    """
    Locate tables that likely contain goalie stats. Strategy:
      1) Look for headings matching "Goalkeepers"/"Målvakter", and take the next table(s).
      2) Fall back to tables with known ids/classes.
    Returns a list of BS4 <table> elements.
    """
    tables: List[BeautifulSoup] = []

    # 1) Headings then following table
    head_re = re.compile(KNOWN_SELECTORS["goalie_section_head_regex"], flags=re.I)
    for heading in doc.find_all(re.compile(r"^h[1-6]$")):
        if head_re.search(heading.get_text(" ", strip=True)):
            # Grab the next table sibling(s)
            nxt = heading.find_next("table")
            if nxt:
                tables.append(nxt)

    # 2) Fallback direct selectors
    for sel in KNOWN_SELECTORS["goalie_table_selectors"]:
        # Handle ':contains()' pseudo (same helper as above)
        if ":contains(" in sel:
            m = re.match(r"^\s*([a-zA-Z0-9\#\.\-:_]+)\s*:\s*contains\(\s*'([^']+)'\s*\)\s*$", sel)
            if m:
                tag_query, literal = m.group(1), m.group(2)
                for tb in doc.select(tag_query):
                    if tb and literal in tb.get_text(" ", strip=True):
                        if tb.name.lower() == "table":
                            tables.append(tb)
                continue
        for tb in doc.select(sel):
            if tb.name.lower() == "table":
                tables.append(tb)

    # Deduplicate by object id
    uniq = []
    seen = set()
    for t in tables:
        if id(t) not in seen:
            uniq.append(t)
            seen.add(id(t))
    return uniq


def parse_goalie_rows_from_table(tb: BeautifulSoup, team_hint: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parse a single goalie table into rows (dicts). This function is defensive because columns differ across sites.
    We look for any of these data points per row:
      - Goalie name
      - Saves
      - Shots against (aka "shots", "on target", etc.)
      - Goals against
      - Save %
      - Time on ice (TOI)
    """
    rows = []
    headers = [th.get_text(" ", strip=True).lower() for th in tb.select("thead th")] or \
              [th.get_text(" ", strip=True).lower() for th in tb.select("tr th")]

    # Build column index map by fuzzy matching common header names (multi-language tolerant)
    def find_idx(words: Iterable[str]) -> Optional[int]:
        for i, h in enumerate(headers):
            for w in words:
                if w in h:
                    return i
        return None

    idx_name = find_idx(["name", "spelare", "målvakt", "målvakter", "goal", "keeper", "goalkeeper"])
    idx_saves = find_idx(["saves", "räddningar", "raddningar"])
    idx_shots = find_idx(["shots", "skott", "on target", "skott mot", "shots against"])
    idx_ga    = find_idx(["ga", "goals against", "insläppta", "insl\u00E4ppta", "inslapp"])
    idx_pct   = find_idx(["%", "save %", "sv%", "sv-proc", "sv%"])
    idx_toi   = find_idx(["toi", "time", "speltid", "minuter", "min", "tid"])

    # Identify body rows
    body_rows = tb.select("tbody tr") or tb.select("tr")
    for tr in body_rows:
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue

        def get_cell(idx: Optional[int]) -> Optional[str]:
            if idx is None:
                return None
            if idx < len(tds):
                return tds[idx].get_text(" ", strip=True)
            return None

        name = get_cell(idx_name) or None
        if not name:
            # If no name, likely a summary row (e.g., "Team Total"). Skip.
            continue

        # Extract numeric fields
        raw_saves = get_cell(idx_saves)
        raw_shots = get_cell(idx_shots)
        raw_ga    = get_cell(idx_ga)
        raw_pct   = get_cell(idx_pct)
        raw_toi   = get_cell(idx_toi)

        # Try to parse (saves/shots) format like "12/15"
        saves_val = safe_int(raw_saves)
        shots_val = safe_int(raw_shots)
        ga_val    = safe_int(raw_ga)

        # If one of the headers held combined "saves/shots" in the saves column:
        if saves_val is not None and raw_saves and "/" in raw_saves and shots_val is None:
            try:
                pair = [int(x) for x in re.findall(r"\d+", raw_saves)]
                if len(pair) >= 2:
                    saves_val, shots_val = pair[0], pair[1]
            except Exception:
                pass

        # If shots is missing but we have saves + GA, derive shots = saves + GA
        if shots_val is None and (saves_val is not None or ga_val is not None):
            if saves_val is not None and ga_val is not None:
                shots_val = saves_val + ga_val

        # Parse save percentage (may be like "86.7", "86,7", "86.7 %")
        pct_val: Optional[float] = None
        if raw_pct:
            pct_txt = raw_pct.replace(",", ".")
            m = re.search(r"(\d+(?:\.\d+)?)", pct_txt)
            if m:
                try:
                    pct_number = float(m.group(1))
                    # If value looks like 0..1, keep; if 0..100, convert
                    pct_val = pct_number if pct_number <= 1.5 else pct_number / 100.0
                except Exception:
                    pct_val = None

        # As a fallback, compute save% from saves/shots
        if pct_val is None:
            pct_val = compute_save_pct(saves_val, shots_val)

        toi_seconds = time_str_to_seconds(raw_toi)

        rows.append({
            "goalie_name": name,
            "team_name": team_hint,
            "saves": saves_val,
            "shots_against": shots_val,
            "goals_against": ga_val,
            "save_pct": pct_val,
            "toi_seconds": toi_seconds,
        })

    return rows


def parse_game_page(session: requests.Session, url: str) -> Tuple[Game, List[Appearance]]:
    """
    Parse a single game page:
      - Extract meta (teams, date)
      - Extract goalie tables for both teams (handles multiple goalies = in-game swaps)
    Returns a Game object and a list of Appearance objects.
    """
    doc = fetch_html(session, url)
    game_id = parse_game_id_from_url(url)

    if doc is None:
        # Return a minimal game object so caller can record a "failed parse"
        logger.warning("Game page unavailable for %s", url)
        return Game(game_id=game_id, url=url, date=None, home_team=None, away_team=None), []

    home_team, away_team, date_dt = parse_game_meta(doc)

    # Find goalie tables (often 1 per team; if combined, both parsed)
    tables = find_goalie_tables(doc)
    appearances: List[Appearance] = []

    # Heuristic: try to split tables per team based on proximity to team names in DOM text
    full_text = doc.get_text(" ", strip=True)
    team_candidates = [t for t in [home_team, away_team] if t]

    for tb in tables:
        # Rough team hint: if team name occurs near this table in the text
        team_hint = None
        local_text = tb.get_text(" ", strip=True)
        for t in team_candidates:
            if t and (t in local_text):
                team_hint = t
                break

        # If no direct hint, leave None; the parser still keeps the rows.
        rows = parse_goalie_rows_from_table(tb, team_hint)
        for r in rows:
            appearances.append(Appearance(
                game_id=game_id,
                goalie_name=r["goalie_name"],
                team_name=r.get("team_name") or "",  # None -> empty string for DB consistency
                saves=r.get("saves"),
                shots_against=r.get("shots_against"),
                goals_against=r.get("goals_against"),
                save_pct=r.get("save_pct"),
                toi_seconds=r.get("toi_seconds"),
            ))

    return Game(game_id=game_id, url=url, date=date_dt, home_team=home_team, away_team=away_team), appearances


# ----------------------------
# Database
# ----------------------------

def init_db(conn: sqlite3.Connection) -> None:
    """
    Create tables if not exist. Keep schema simple & normalized.
    """
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS games (
        game_id TEXT PRIMARY KEY,
        url TEXT,
        date_utc TEXT,              -- ISO8601
        home_team TEXT,
        away_team TEXT,
        scraped_at_utc TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS goalies (
        goalie_id INTEGER PRIMARY KEY AUTOINCREMENT,
        goalie_name TEXT UNIQUE
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS appearances (
        appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT,
        goalie_id INTEGER,
        team_name TEXT,
        saves INTEGER,
        shots_against INTEGER,
        goals_against INTEGER,
        save_pct REAL,         -- 0..1
        toi_seconds INTEGER,
        FOREIGN KEY (game_id) REFERENCES games(game_id),
        FOREIGN KEY (goalie_id) REFERENCES goalies(goalie_id)
    )
    """)
    # Useful indexes for speed
    cur.execute("CREATE INDEX IF NOT EXISTS idx_appearances_goalie ON appearances(goalie_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_appearances_game ON appearances(game_id)")
    conn.commit()


def upsert_game(conn: sqlite3.Connection, game: Game) -> None:
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO games (game_id, url, date_utc, home_team, away_team, scraped_at_utc)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(game_id) DO UPDATE SET
        url=excluded.url,
        date_utc=excluded.date_utc,
        home_team=excluded.home_team,
        away_team=excluded.away_team,
        scraped_at_utc=excluded.scraped_at_utc
    """, (
        game.game_id,
        game.url,
        game.date.isoformat() if game.date else None,
        game.home_team,
        game.away_team,
        datetime.utcnow().isoformat(timespec="seconds")
    ))
    conn.commit()


def get_or_create_goalie_id(conn: sqlite3.Connection, goalie_name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT goalie_id FROM goalies WHERE goalie_name = ?", (goalie_name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO goalies (goalie_name) VALUES (?)", (goalie_name,))
    conn.commit()
    return cur.lastrowid


def insert_appearances(conn: sqlite3.Connection, game_id: str, appearances: List[Appearance]) -> None:
    """
    Insert appearances for one game. We do a simple strategy:
     - Delete old rows for that game (idempotent re-runs).
     - Insert fresh appearances.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM appearances WHERE game_id = ?", (game_id,))
    for a in appearances:
        goalie_id = get_or_create_goalie_id(conn, a.goalie_name)
        cur.execute("""
        INSERT INTO appearances
            (game_id, goalie_id, team_name, saves, shots_against, goals_against, save_pct, toi_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a.game_id,
            goalie_id,
            a.team_name,
            a.saves,
            a.shots_against,
            a.goals_against,
            a.save_pct,
            a.toi_seconds
        ))
    conn.commit()


# ----------------------------
# Orchestration
# ----------------------------

def scrape_fixture_for_game_links(session: requests.Session, fixture_url: str) -> List[str]:
    """
    Load the fixture list and return unique game links.
    """
    doc = fetch_html(session, fixture_url)
    if doc is None:
        raise RuntimeError(f"Failed to load fixture list: {fixture_url}")
    links = find_game_links(doc, fixture_url)
    if not links:
        logger.warning("No game links discovered on fixture page. "
                       "Consider adjusting KNOWN_SELECTORS['game_link_hrefs_contains'].")
    return links


def process_games(session: requests.Session, game_urls: List[str], conn: sqlite3.Connection) -> None:
    """
    Fetch and parse all game pages concurrently, with progress and robust error handling.
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(parse_game_page, session, url): url for url in game_urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scraping games", unit="game"):
            url = futures[fut]
            try:
                game, appearances = fut.result()
                upsert_game(conn, game)
                insert_appearances(conn, game.game_id, appearances)
            except Exception as e:
                logger.error("Error processing %s: %s", url, e)


# ----------------------------
# Reporting / Graphing
# ----------------------------

def build_goalie_timeline(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Build a DataFrame with per-appearance save% over time for every goalie.
    Each row corresponds to ONE appearance (i.e., if a goalie swaps mid-game, that's a separate row).
    """
    q = """
    SELECT
        g.goalie_name,
        gm.date_utc,
        ap.game_id,
        ap.save_pct,
        ap.saves,
        ap.shots_against,
        ap.goals_against,
        ap.toi_seconds,
        ap.team_name
    FROM appearances ap
    JOIN goalies g  ON g.goalie_id = ap.goalie_id
    JOIN games gm   ON gm.game_id   = ap.game_id
    WHERE gm.date_utc IS NOT NULL
    ORDER BY g.goalie_name, gm.date_utc, ap.appearance_id
    """
    df = pd.read_sql_query(q, conn, parse_dates=["date_utc"])
    # Defensive: there may be rows with NULL save_pct; we keep them (plotted as gaps)
    return df


def plot_savepct_timeline(df: pd.DataFrame, outfile: str, max_goalies: Optional[int] = MAX_GOALIES_IN_GRAPH) -> None:
    """
    Plot a timeline of save% per goalie.
    - One line per goalie (with markers for appearances).
    - If too many goalies, optionally cap to top-N by number of appearances (to keep the chart readable).
    """
    if df.empty:
        logger.warning("No data to plot.")
        return

    # Decide which goalies to include
    counts = df.groupby("goalie_name")["game_id"].nunique().sort_values(ascending=False)
    if max_goalies is not None and len(counts) > max_goalies:
        kept_goalies = set(counts.head(max_goalies).index)
        df_plot = df[df["goalie_name"].isin(kept_goalies)].copy()
        dropped = len(counts) - len(kept_goalies)
        notice = f"(showing top {len(kept_goalies)} goalies by appearances; {dropped} hidden)"
    else:
        df_plot = df.copy()
        notice = None

    # Sort by date for plotting
    df_plot = df_plot.sort_values(["goalie_name", "date_utc"])

    plt.figure(figsize=(14, 8))
    # Plot each goalie as its own series, connecting their appearances over time
    for goalie, sub in df_plot.groupby("goalie_name"):
        # Sort for safety
        sub = sub.sort_values("date_utc")
        plt.plot(sub["date_utc"], sub["save_pct"], marker="o", linewidth=1.5, label=goalie)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.title("Floorball Goalies — Save Percentage Timeline")
    plt.xlabel("Game Date")
    plt.ylabel("Save Percentage")
    plt.grid(True, linestyle="--", alpha=0.4)
    # Place legend outside to avoid clutter
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title="Goalie")
    if notice:
        plt.suptitle(notice, y=0.98, fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    logger.info("Saved timeline plot → %s", outfile)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    session = build_session()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # 1) Discover game links from the fixture list
    try:
        game_urls = scrape_fixture_for_game_links(session, FIXTURE_URL)
        if not game_urls:
            logger.error("No game URLs found — cannot proceed.")
            return 2
    except Exception as e:
        logger.error("Failed to scrape fixture list: %s", e)
        return 2

    # 2) Parse & store all games/appearances (concurrently)
    process_games(session, game_urls, conn)

    # 3) Build DF and plot
    df = build_goalie_timeline(conn)
    plot_savepct_timeline(df, OUTPUT_FIG, max_goalies=MAX_GOALIES_IN_GRAPH)

    # 4) Simple console report (optional)
    if not df.empty:
        summary = (
            df.groupby("goalie_name")
              .agg(
                  appearances=("game_id", "nunique"),
                  avg_sv_pct=("save_pct", "mean")
              )
              .sort_values(["appearances", "avg_sv_pct"], ascending=[False, False])
        )
        # Console printout (kept concise; DB + PNG is the main output)
        print("\n=== Summary (per goalie) ===")
        print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nFigure saved at: {OUTPUT_FIG}")
        print(f"Database saved at: {DB_PATH}")
    else:
        print("No appearance data parsed. Check selectors/patterns and try again.")
    conn.close()
    return 0


if __name__ == "__main__":
    # Exit code for shell integration
    sys.exit(main())
