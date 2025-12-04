"""Display helpers for the innebandy goalie statistics scraper.

This module focuses on presentation concerns: summarising datasets, building
interactive Plotly figures, and writing HTML/Excel outputs. Fixture URLs are
configured directly in code so the script can be executed without command-line
arguments.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import fetch_data

logger = logging.getLogger(__name__)


# URLs are hard-coded to satisfy the "no arguments" requirement. Adjust the
# list below to fetch and display data from other leagues.
LEAGUE_SOURCES: List[Tuple[str, str]] = [
    ("Primary league", fetch_data.DEFAULT_FIXTURE_URL),
    ("Secondary league", fetch_data.ALTERNATE_FIXTURE_URL),
]

DEFAULT_EXCEL_OUTPUT = fetch_data.DEFAULT_OUTPUT
DEFAULT_PLOT_OUTPUT = fetch_data.DEFAULT_PLOT
DEFAULT_DASHBOARD = Path("index.html")
DEFAULT_TEMPLATE = Path(__file__).with_name("dashboard_template.html")


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
    stats = {
        "total_goalies": summary["goalie"].nunique(),
        "total_teams": summary["team"].nunique(),
        "date_range": _date_range_text(cumulative),
        "zero_shot_goalies": len(dropped_goalies),
    }

    goalie_to_team = (
        summary.set_index("goalie")["team"].dropna().fillna("Unknown").to_dict()
    )
    team_options = sorted(summary["team"].fillna("Unknown").unique().tolist())
    summary_records = summary.to_dict(orient="records")

    import plotly.io as pio

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

    if not DEFAULT_TEMPLATE.exists():
        raise FileNotFoundError(
            f"Dashboard template not found at {DEFAULT_TEMPLATE.resolve()}"
        )

    template = DEFAULT_TEMPLATE.read_text(encoding="utf-8")

    return template.replace("__TEAM_OPTIONS__", dropdown_options).replace(
        "__LEAGUES_JSON__", league_json
    )


def generate_outputs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    league_snapshots: List[LeagueSnapshot] = []
    for name, url in LEAGUE_SOURCES:
        logger.info("Fetching league '%s' from %s", name, url)
        games_df, appearances_df = fetch_data.scrape_data(url)
        cumulative, dropped_goalies = fetch_data.prepare_cumulative_save_percentages(
            games_df, appearances_df
        )
        figure = build_figure(cumulative, dropped_goalies)
        summary = summarise_goalies(cumulative)
        league_snapshots.append(
            _build_league_snapshot(name, figure, cumulative, summary, dropped_goalies)
        )

        if name == LEAGUE_SOURCES[0][0]:
            fetch_data.write_excel(games_df, appearances_df, DEFAULT_EXCEL_OUTPUT)
            with contextlib.suppress(Exception):
                fetch_data.create_plot(appearances_df, DEFAULT_PLOT_OUTPUT)

    dashboard_html = build_dashboard_html(league_snapshots)
    DEFAULT_DASHBOARD.write_text(dashboard_html, encoding="utf-8")
    logger.info("Dashboard written to %s", DEFAULT_DASHBOARD.resolve())


if __name__ == "__main__":
    generate_outputs()
