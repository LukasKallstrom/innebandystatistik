import pytest

from innebandy.aggregator import summarize_goalies
from innebandy.models import GoalieAppearance


def make_appearance(goalie, team, saves, goals, game_id):
    return GoalieAppearance(
        game_id=game_id,
        goalie=goalie,
        team=team,
        opponent="",
        saves=saves,
        shots_against=saves + goals,
        goals_against=goals,
        time_on_ice_seconds=60 * 60,
    )


def test_summary_collapses_multiple_games():
    appearances = [
        make_appearance("Alice", "Falcons", saves=20, goals=2, game_id="G1"),
        make_appearance("Alice", "Falcons", saves=18, goals=1, game_id="G2"),
        make_appearance("Bob", "Wolves", saves=10, goals=0, game_id="G3"),
    ]

    summaries = summarize_goalies(appearances)

    alice = next(summary for summary in summaries if summary.goalie == "Alice")
    assert alice.games == 2
    assert alice.saves == 38
    assert alice.shots_against == 41
    assert alice.goals_against == 3
    assert alice.time_on_ice_seconds == 2 * 60 * 60
    assert alice.save_percentage == pytest.approx(38 / 41, rel=1e-3)

    bob = next(summary for summary in summaries if summary.goalie == "Bob")
    assert bob.games == 1
    assert bob.save_percentage == 1.0
