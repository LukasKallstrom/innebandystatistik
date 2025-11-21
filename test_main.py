# test_main.py
import re
import pytest

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency in test env
    pytest.skip("beautifulsoup4 is required for these tests", allow_module_level=True)

from main import parse_game, Appearance


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
    # Två rader (två tabeller, en per lag)
    assert len(apps) == 2
    # Kontrollera första
    a0 = next(a for a in apps if a.goalie_name == "A Keeper")
    assert a0.shots_against == 18
    assert a0.goals_against == 2
    assert a0.saves == 16  # härleds från 18-2 om 'Räddningar' ej finns
    assert abs(a0.save_pct - 0.88) < 1e-9
    a1 = next(a for a in apps if a.goalie_name == "B Keeper")


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
