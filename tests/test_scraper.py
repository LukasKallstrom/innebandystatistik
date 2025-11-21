from pathlib import Path

import pytest

from innebandy.scraper import collect_goalie_stats, discover_game_links


def test_discover_game_links_filters_and_deduplicates():
    html = """
    <html>
      <body>
        <a href="/game?id=1&scr=game">Game 1</a>
        <a href="/other">Other</a>
        <a href="/game?id=1&scr=game">Duplicate</a>
        <a href="result?id=2&scr=result">Game 2</a>
      </body>
    </html>
    """

    links = discover_game_links(html, base_url="http://example.com/fixtures")

    assert links == [
        "http://example.com/game?id=1&scr=game",
        "http://example.com/result?id=2&scr=result",
    ]


def test_collect_goalie_stats_reads_fixture_and_game_pages(tmp_path: Path):
    fixture = tmp_path / "fixtures.html"
    game1 = tmp_path / "game1.html"
    game2 = tmp_path / "game2.html"

    game1.write_text(
        """
        <table>
          <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th></tr>
          <tr><td>Alice</td><td>Falcons</td><td>20</td><td>2</td><td>22</td></tr>
        </table>
        """
    )
    game2.write_text(
        """
        <table>
          <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th></tr>
          <tr><td>Bob</td><td>Wolves</td><td>15</td><td>3</td><td>18</td></tr>
        </table>
        """
    )
    fixture.write_text(
        f'<a href="{game1.name}?scr=game">Game 1</a>'
        f'<a href="{game2.name}?scr=result">Game 2</a>'
    )

    fixture_url = fixture.resolve().as_uri()

    appearances = collect_goalie_stats(fixture_url)

    assert len(appearances) == 2
    assert {a.goalie for a in appearances} == {"Alice", "Bob"}
    assert {a.game_id for a in appearances} == {
        game1.name.replace(".html", "") + "_scr_game",
        game2.name.replace(".html", "") + "_scr_result",
    }


def test_collect_goalie_stats_errors_when_no_games(tmp_path: Path):
    fixture = tmp_path / "fixtures.html"
    fixture.write_text("<html><body>No links</body></html>")

    with pytest.raises(SystemExit):
        collect_goalie_stats(fixture.resolve().as_uri())
