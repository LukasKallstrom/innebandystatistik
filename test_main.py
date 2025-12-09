# test_main.py
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from fetch_data import (
    Appearance,
    _extract_date_text,
    _inject_shots_from_events,
    parse_game,
    parse_game_id,
    prepare_cumulative_save_percentages,
)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def test_player_table_fallback_pos_mv_parsing():
    """Röktest: Spelartabell ('Statistik i matchen') med Pos.=MV.
    Ska filtrera fram endast målvakt och räkna saves = skott - insl.mål.
    Procent med komma ska tolkas korrekt.
    """
    html = """
    <html><body>
      <span id="ctl00_PlaceHolderMain_lblHomeTeam">FBC Partille</span>
      <span id="ctl00_PlaceHolderMain_lblAwayTeam">IBK Lidköping</span>
      <span id="ctl00_PlaceHolderMain_lblMatchDate">2022-10-01 16:00</span>

      <!-- Hemmalagets spelartabell -->
      <table>
        <thead>
          <tr>
            <th>Nr</th><th>Namn</th><th>Pos.</th><th>Skott</th><th>Insl.Mål</th><th>Räddn.(%)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>1</td><td>Erik Keep</td><td>MV</td><td>31</td><td>3</td><td>90,3 %</td></tr>
          <tr><td>24</td><td>Utespelare</td><td>B</td><td>5</td><td>—</td><td>—</td></tr>
        </tbody>
      </table>

      <!-- Bortalagets spelartabell -->
      <table>
        <thead>
          <tr>
            <th>Nr</th><th>Namn</th><th>Pos.</th><th>Skott</th><th>Insl.Mål</th><th>Räddn.(%)</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>30</td><td>Ida Net</td><td>Målvakt</td><td>22</td><td>6</td><td>72,7%</td></tr>
          <tr><td>9</td><td>Forward</td><td>F</td><td>4</td><td>—</td><td>—</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    doc = _soup(html)
    game, apps = parse_game(doc, url="http://statistik.innebandy.se/ft.aspx?scr=result&fmid=1570524")

    # Metadata
    assert game.home_team == "FBC Partille"
    assert game.away_team == "IBK Lidköping"
    assert isinstance(game.date, datetime)

    # Två målvakter, en per lag
    assert len(apps) == 2
    # Hemma
    a_home = next(a for a in apps if a.team_name == "FBC Partille")
    assert a_home.goalie_name == "Erik Keep"
    assert a_home.shots_against == 31
    assert a_home.goals_against == 3
    assert a_home.saves == 28
    assert a_home.save_pct is not None
    # procentsats "90,3 %" -> 0.903 (tolerera flytpunktsavrundning)
    assert abs(a_home.save_pct - 0.903) < 1e-6

    # Borta
    a_away = next(a for a in apps if a.team_name == "IBK Lidköping")
    assert a_away.goalie_name == "Ida Net"
    assert a_away.shots_against == 22
    assert a_away.goals_against == 6
    assert a_away.saves == 16
    # "72,7%" -> 0.727
    assert abs(a_away.save_pct - 0.727) < 1e-6

    # TOI saknas i dessa tabeller -> None
    assert a_home.time_on_ice_seconds is None
    assert a_away.time_on_ice_seconds is None


def test_explicit_goalie_table_path_kept():
    """Röktest: Explicit rubrik 'Målvakter' och separat tabell.
    Här ska den ursprungliga vägen (extract_goalie_tables) fungera.
    """
    html = """
    <html><body>
      <span id="ctl00_PlaceHolderMain_lblHomeTeam">Lag A</span>
      <span id="ctl00_PlaceHolderMain_lblAwayTeam">Lag B</span>
      <span id="ctl00_PlaceHolderMain_lblMatchDate">2021-11-20</span>

      <h3>Målvakter</h3>
      <table>
        <thead>
          <tr>
            <th>Spelare</th><th>Skott</th><th>Insläppta</th><th>Rädd%</th><th>Tid</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>A Keeper</td><td>18</td><td>2</td><td>88%</td><td>60:00</td></tr>
        </tbody>
      </table>

      <h3>Målvakter</h3>
      <table>
        <thead>
          <tr>
            <th>Spelare</th><th>Skott</th><th>Insläppta</th><th>Rädd%</th><th>Tid</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>B Keeper</td><td>25</td><td>5</td><td>80%</td><td>59:12</td></tr>
        </tbody>
      </table>
    </body></html>
    """
    doc = _soup(html)
    game, apps = parse_game(doc, url="http://example.test/game?fmid=ABC123")

    assert game.home_team == "Lag A"
    assert game.away_team == "Lag B"
    assert game.date.year == 2021

    # Två rader (två tabeller, en per lag)
    assert len(apps) == 2
    # Kontrollera första
    a0 = next(a for a in apps if a.goalie_name == "A Keeper")
    assert a0.shots_against == 18
    assert a0.goals_against == 2
    assert a0.saves == 16  # härleds från 18-2 om 'Räddningar' ej finns
    assert abs(a0.save_pct - 0.88) < 1e-9
    # Tid "60:00" -> 3600
    assert a0.time_on_ice_seconds == 3600

    a1 = next(a for a in apps if a.goalie_name == "B Keeper")
    assert a1.time_on_ice_seconds == 59 * 60 + 12


def test_percent_with_spaces_and_nbsp_is_parsed():
    """Röktest: procentsats med hårt blanksteg/extra mellanslag."""
    html = """
    <html><body>
      <span id="ctl00_PlaceHolderMain_lblHomeTeam">Hemma</span>
      <span id="ctl00_PlaceHolderMain_lblAwayTeam">Borta</span>
      <table>
        <thead>
          <tr>
            <th>Namn</th><th>Pos.</th><th>Skott</th><th>Insl.Mål</th><th>Räddn.(%)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Keeper X</td><td>MV</td><td>10</td><td>1</td>
            <td>90\u00a0%</td>  <!-- NBSP före procent -->
          </tr>
        </tbody>
      </table>
    </body></html>
    """
    doc = _soup(html)
    game, apps = parse_game(doc, url="http://example.test/x?fmid=Z")
    assert len(apps) == 1
    a = apps[0]
    assert a.saves == 9
    # 90 % -> 0.9
    assert abs(a.save_pct - 0.9) < 1e-9


def test_matchinformation_date_beats_comment_block():
    """Date in Matchinformation should be used instead of commented metadata."""

    html = """
    <html><body>
      <!-- hidden metadata with different kickoff time -->
      <!-- <matchtid>2025-09-20 02:00</matchtid> -->

      <div class="clMatchView">
        <table class="clCommonGrid" id="iMatchInfo" cellspacing="0">
          <tbody>
            <tr><td>Matchnummer</td><td>123</td></tr>
            <tr><td>Tävling</td><td>SSL</td></tr>
            <tr><td>Tid</td><td><span>2025-09-20<!-- br ok --> 14:00</span></td></tr>
          </tbody>
        </table>
      </div>
    </body></html>
    """

    doc = _soup(html)
    game, _ = parse_game(doc, url="http://example.test/game?fmid=ID")

    assert isinstance(game.date, datetime)
    assert game.date == datetime(2025, 9, 20, 14, 0)


def test_matchtid_comment_used_when_no_visible_date():
    """Hidden <matchtid> in HTML comments should be respected when tables are missing."""
    html = """
    <html><body>
      <!-- <matchtid>2024-02-10 18:30</matchtid> -->
      <div>Ingen tabell här</div>
    </body></html>
    """

    doc = _soup(html)
    assert _extract_date_text(doc) == "2024-02-10 18:30"

    game, apps = parse_game(doc, url="http://example.test/game?GameID=XYZ-789")
    assert game.date == datetime(2024, 2, 10, 18, 30)
    assert parse_game_id(game.url) == "XYZ-789"
    assert apps == []


def test_event_driven_allocation_distributes_shots_and_goals():
    """Shots/goals should be allocated to goalies based on substitution timeline."""
    html = """
    <html><body>
      <span id="Skottstatistik">Skottstatistik: 18 - 15 (9 - 5, 6 - 7, 3 - 3)</span>
      <table class="clTblMatch">
        <tr class="clPeriodStart"><td>Period 1</td></tr>
        <tr><td>Målvaktsbyte</td><td>00:00</td><td></td><td>Goalie A</td><td>Team Away</td></tr>
        <tr><td>Mål</td><td>10:00</td><td></td><td>Forward</td><td>Team Home</td></tr>
        <tr class="clPeriodStart"><td>Period 2</td></tr>
        <tr><td>Mål</td><td>00:40</td><td></td><td>Forward</td><td>Team Home</td></tr>
        <tr><td>Målvaktsbyte</td><td>05:00</td><td></td><td>Goalie B</td><td>Team Away</td></tr>
        <tr class="clPeriodStart"><td>Period 3</td></tr>
        <tr><td>Mål</td><td>05:00</td><td></td><td>Forward</td><td>Team Home</td></tr>
      </table>
    </body></html>
    """
    doc = _soup(html)
    appearances = [
        Appearance(
            game_id="match1",
            goalie_name="Home Goalie",
            team_name="Team Home",
            saves=15,
            shots_against=18,
            goals_against=3,
            save_pct=0.83,
            time_on_ice_seconds=3600,
        ),
        Appearance(
            game_id="match1",
            goalie_name="Goalie A",
            team_name="Team Away",
            saves=None,
            shots_against=None,
            goals_against=None,
            save_pct=None,
            time_on_ice_seconds=None,
        ),
        Appearance(
            game_id="match1",
            goalie_name="Goalie B",
            team_name="Team Away",
            saves=None,
            shots_against=None,
            goals_against=None,
            save_pct=None,
            time_on_ice_seconds=None,
        ),
    ]

    updated = _inject_shots_from_events(
        doc, home="Team Home", away="Team Away", game_id="match1", appearances=appearances
    )

    goalie_a = next(app for app in updated if app.goalie_name == "Goalie A")
    goalie_b = next(app for app in updated if app.goalie_name == "Goalie B")

    assert goalie_a.shots_against == 10
    assert goalie_a.goals_against == 2
    assert goalie_a.saves == 8
    assert abs(goalie_a.save_pct - 0.8) < 1e-9
    assert goalie_a.time_on_ice_seconds == 1500  # 25 minutes

    assert goalie_b.shots_against == 8
    assert goalie_b.goals_against == 1
    assert goalie_b.saves == 7
    assert abs(goalie_b.save_pct - 7 / 8) < 1e-9
    assert goalie_b.time_on_ice_seconds == 2100  # 35 minutes


def test_prepare_cumulative_rebuilds_missing_counts_and_filters_zero_shots():
    games = pd.DataFrame(
        [
            {"game_id": "g1", "url": "u1", "date": datetime(2024, 1, 1), "home_team": "A", "away_team": "B"},
            {"game_id": "g2", "url": "u2", "date": pd.NaT, "home_team": "A", "away_team": "B"},
        ]
    )
    appearances = pd.DataFrame(
        [
            {"game_id": "g1", "goalie": "Goalie A", "team": "A", "saves": 28, "shots_against": pd.NA, "goals_against": 2, "save_pct": pd.NA},
            {"game_id": "g2", "goalie": "Goalie A", "team": "A", "saves": pd.NA, "shots_against": 25, "goals_against": 1, "save_pct": pd.NA},
            {"game_id": "g1", "goalie": "Goalie B", "team": "B", "saves": 0, "shots_against": 0, "goals_against": 0, "save_pct": 0},
        ]
    )

    cumulative, dropped = prepare_cumulative_save_percentages(games, appearances)

    assert dropped == ["Goalie B"]
    assert list(cumulative["goalie"].unique()) == ["Goalie A"]
    assert len(cumulative) == 2

    # Synthetic date for g2 should be earlier than real date for g1
    assert cumulative.iloc[0]["game_id"] == "g2"
    assert cumulative.iloc[1]["game_id"] == "g1"
    assert cumulative.iloc[0]["date"] < cumulative.iloc[1]["date"]

    # Reconstructed counts and cumulative percentages
    row_g2 = cumulative[cumulative["game_id"] == "g2"].iloc[0]
    row_g1 = cumulative[cumulative["game_id"] == "g1"].iloc[0]

    assert row_g2["shots_against"] == 25
    assert row_g2["saves"] == 24
    assert abs(row_g2["cumulative_save_pct"] - 0.96) < 1e-9

    assert row_g1["shots_against"] == 30  # 28 + 2
    assert row_g1["saves"] == 28
    assert abs(row_g1["cumulative_save_pct"] - (52 / 55)) < 1e-9
