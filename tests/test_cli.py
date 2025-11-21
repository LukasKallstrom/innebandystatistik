import json
import subprocess
import sys
from pathlib import Path

import pytest

from main import run


HTML_TEMPLATE = """
<table>
  <tr><th>Goalie</th><th>Team</th><th>Saves</th><th>Goals Against</th><th>Shots Against</th></tr>
  <tr><td>{goalie}</td><td>{team}</td><td>{saves}</td><td>{goals}</td><td>{shots}</td></tr>
</table>
"""


def write_game(tmp_path: Path, name: str, goalie: str, team: str, saves: int, goals: int):
    html = HTML_TEMPLATE.format(goalie=goalie, team=team, saves=saves, goals=goals, shots=saves + goals)
    html_path = tmp_path / f"{name}.html"
    html_path.write_text(html)
    return html_path


def test_run_function_writes_expected_json(tmp_path):
    game1 = write_game(tmp_path, "g1", "Alice", "Falcons", saves=20, goals=2)
    game2 = write_game(tmp_path, "g2", "Bob", "Wolves", saves=15, goals=3)
    fixture = tmp_path / "fixtures.html"
    fixture.write_text(
        f'<a href="{game1.name}?scr=game">Game 1</a>' f'<a href="{game2.name}?scr=result">Game 2</a>'
    )

    output_path = tmp_path / "out.json"
    payload = run(fixture.resolve().as_uri(), output_path)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == payload
    assert len(payload["appearances"]) == 2
    assert payload["summaries"][0]["games"] == 1


def test_cli_exit_on_invalid_fixture(tmp_path):
    fixture = tmp_path / "fixtures.html"
    fixture.write_text("<html><body>No links</body></html>")
    output_path = tmp_path / "out.json"

    process = subprocess.run(
        [sys.executable, "main.py", "--fixture-url", fixture.as_uri(), "--output", str(output_path)],
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert "No game links" in process.stderr or "No game links" in process.stdout
    assert not output_path.exists()
