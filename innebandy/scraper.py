"""Utilities for downloading goalie statistics directly from the fixture list site."""
from __future__ import annotations

import logging
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib.parse import urljoin, urlparse, unquote

from .models import GoalieAppearance
from .parser import ParsingError, parse_goalie_table

logger = logging.getLogger(__name__)

GAME_LINK_KEYWORDS = ("scr=game", "scr=result", "matchid=", "fmid=", "gameid=")


def _clean_game_id(url: str) -> str:
    parsed = urlparse(url)
    path_part = parsed.path.split("/")[-1]
    if "." in path_part:
        path_part = path_part.rsplit(".", 1)[0]
    candidate = f"{path_part}?{parsed.query}".strip("?")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", candidate).strip("_")
    return cleaned[-80:] or "unknown_game"


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def discover_game_links(fixture_html: str, *, base_url: str) -> List[str]:
    """Extract candidate game URLs from the fixture list HTML."""

    extractor = _LinkExtractor()
    extractor.feed(fixture_html)
    absolute_links: List[str] = []
    for href in extractor.links:
        lowered = href.lower()
        if any(keyword in lowered for keyword in GAME_LINK_KEYWORDS):
            absolute_links.append(urljoin(base_url, href))
    # Preserve order but remove duplicates
    seen: set[str] = set()
    deduped: List[str] = []
    for link in absolute_links:
        if link not in seen:
            deduped.append(link)
            seen.add(link)
    return deduped


def default_fetch(url: str) -> str:
    """Fetch *url* using urllib with a browser-like user agent."""

    parsed = urlparse(url)
    if parsed.scheme == "file":
        local_path = Path(unquote(parsed.path))
        return local_path.read_text()

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:  # type: ignore[arg-type]
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def collect_goalie_stats(
    fixture_url: str,
    *,
    fetcher=default_fetch,
) -> List[GoalieAppearance]:
    """Download all game pages referenced by *fixture_url* and parse goalie stats."""

    fixture_html = fetcher(fixture_url)
    game_links = discover_game_links(fixture_html, base_url=fixture_url)
    if not game_links:
        raise SystemExit("No game links discovered in the supplied fixture page")

    appearances = []
    for link in game_links:
        html = fetcher(link)
        game_id = _clean_game_id(link)
        try:
            appearances.extend(parse_goalie_table(html, game_id=game_id))
        except ParsingError as exc:
            logger.warning("Skipping game %s: %s", game_id, exc)
    if not appearances:
        raise SystemExit("No goalie appearances could be parsed from any game page")
    return appearances

