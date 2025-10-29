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

    idx_name = find_idx(["nam]()
