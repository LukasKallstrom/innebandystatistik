import pytest

from innebandy.parser import ParsingError, parse_goalie_table


def test_parses_basic_table():
    html = """
    <table>
      <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th><th>TOI</th></tr>
      <tr><td>Alice Netminder</td><td>Falcons</td><td>20</td><td>2</td><td>22</td><td>60:00</td></tr>
      <tr><td>Bob Blocker</td><td>Wolves</td><td>18</td><td>4</td><td>22</td><td>59:12</td></tr>
    </table>
    """

    appearances = parse_goalie_table(html, game_id="G1", home_team="Falcons", away_team="Wolves")

    assert len(appearances) == 2
    first = appearances[0]
    assert first.goalie == "Alice Netminder"
    assert first.team == "Falcons"
    assert first.opponent == "Wolves"
    assert first.saves == 20
    assert first.goals_against == 2
    assert first.shots_against == 22
    assert first.save_percentage == pytest.approx(0.909)
    assert first.time_on_ice_seconds == 3600


def test_derives_shots_against_when_missing_column():
    html = """
    <table>
      <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th></tr>
      <tr><td>Alice Netminder</td><td>Falcons</td><td>12</td><td>1</td></tr>
    </table>
    """

    [appearance] = parse_goalie_table(html, game_id="G2", home_team="Falcons", away_team="Wolves")
    assert appearance.shots_against == 13
    assert appearance.save_percentage == pytest.approx(0.923)


def test_fills_missing_goals_with_difference():
    html = """
    <table>
      <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th></tr>
      <tr><td>Alice Netminder</td><td>Falcons</td><td>15</td><td></td><td>16</td></tr>
    </table>
    """

    [appearance] = parse_goalie_table(html, game_id="G3", home_team="Falcons", away_team="Wolves")
    assert appearance.goals_against == 1
    assert appearance.shots_against == 16
    assert appearance.save_percentage == pytest.approx(0.938)


def test_invalid_time_format_raises():
    html = """
    <table>
      <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th><th>TOI</th></tr>
      <tr><td>Alice Netminder</td><td>Falcons</td><td>10</td><td>0</td><td>10</td><td>12:99</td></tr>
    </table>
    """

    with pytest.raises(ParsingError):
        parse_goalie_table(html, game_id="G4", home_team="Falcons", away_team="Wolves")


def test_requires_team_information():
    html = """
    <table>
      <tr><th>Goalie</th><th>Saves</th><th>Goals Against</th></tr>
      <tr><td>Alice Netminder</td><td>10</td><td>0</td></tr>
    </table>
    """

    with pytest.raises(ParsingError):
        parse_goalie_table(html, game_id="G5", home_team="Falcons", away_team="Wolves")


def test_mismatched_shots_against_raises():
    html = """
    <table>
      <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th></tr>
      <tr><td>Alice Netminder</td><td>Falcons</td><td>10</td><td>1</td><td>20</td></tr>
    </table>
    """

    with pytest.raises(ParsingError):
        parse_goalie_table(html, game_id="G6", home_team="Falcons", away_team="Wolves")
