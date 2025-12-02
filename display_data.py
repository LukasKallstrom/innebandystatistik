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

    return f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Goalie Save Percentage Dashboard</title>
    <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />
    <link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap\" />
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #0f172a;
        --bg-panel: rgba(15, 23, 42, 0.75);
        --bg-light: #f8fafc;
        --border: rgba(148, 163, 184, 0.35);
        --text: #0f172a;
        --text-muted: #475569;
        --accent: #2563eb;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}

      body {{
        margin: 0;
        padding: 0;
        background: linear-gradient(160deg, #020617, #0f172a 55%, #1e293b);
        color: white;
        min-height: 100vh;
      }}

      .page {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 2.5rem 1.5rem 4rem;
        display: flex;
        flex-direction: column;
        gap: 2.5rem;
      }}

      header {{
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }}

      h1 {{
        margin: 0;
        font-size: clamp(1.8rem, 4vw, 2.4rem);
        letter-spacing: -0.02em;
      }}

      p.description {{
        margin: 0;
        color: rgba(255, 255, 255, 0.75);
        max-width: 820px;
        line-height: 1.6;
      }}

      .panel {{
        background: var(--bg-panel);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(10px);
      }}

      .controls {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        align-items: flex-end;
      }}

      label {{
        display: block;
        margin-bottom: 0.35rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 600;
        letter-spacing: 0.01em;
      }}

      select, input {{
        width: 100%;
        padding: 0.65rem 0.85rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background: rgba(255, 255, 255, 0.06);
        color: white;
        font-size: 0.95rem;
        transition: border-color 0.2s ease, background 0.2s ease;
      }}

      select:focus, input:focus {{
        outline: none;
        border-color: rgba(37, 99, 235, 0.8);
        background: rgba(255, 255, 255, 0.1);
      }}

      button {{
        background: linear-gradient(120deg, #2563eb, #4f46e5);
        color: white;
        border: none;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.01em;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.2s ease;
        box-shadow: 0 12px 35px rgba(37, 99, 235, 0.25);
      }}

      button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.32);
      }}

      button:active {{
        transform: translateY(0);
        opacity: 0.9;
      }}

      #savepct-chart {{
        width: 100%;
        height: 580px;
      }}

      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
      }}

      .stat-card {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.9rem 1rem;
      }}

      .stat-label {{
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
      }}

      .stat-value {{
        font-size: 1.4rem;
        font-weight: 700;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
      }}

      th, td {{
        padding: 0.75rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      }}

      th {{
        color: rgba(255, 255, 255, 0.8);
        font-weight: 700;
        cursor: pointer;
        white-space: nowrap;
      }}

      td {{
        color: rgba(255, 255, 255, 0.85);
      }}

      tr:hover td {{
        background: rgba(255, 255, 255, 0.03);
      }}

      .tag {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.85rem;
      }}

      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.2);
        color: white;
        border: 1px solid rgba(37, 99, 235, 0.45);
        font-size: 0.85rem;
      }}

      .muted {{
        color: rgba(255, 255, 255, 0.65);
      }}

      .dropped-list {{
        margin-top: 0.4rem;
        color: rgba(255, 255, 255, 0.75);
        line-height: 1.6;
      }}

      footer {{
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.9rem;
        text-align: center;
        padding-bottom: 2rem;
      }}
    </style>
  </head>
  <body>
    <div class=\"page\">
      <header>
        <h1>Goalie Save Percentage Dashboard</h1>
        <p class=\"description\">Interactive view of goalie cumulative save percentage across leagues. Use the dropdowns below to switch leagues, filter by team, and search goalies.</p>
      </header>

      <div class=\"panel\">
        <div class=\"controls\">
          <div>
            <label for=\"league-selector\">League</label>
            <select id=\"league-selector\"></select>
          </div>
          <div>
            <label for=\"team-filter\">Team filter</label>
            <select id=\"team-filter\">
              <option value=\"ALL\">All teams</option>
              {dropdown_options}
            </select>
          </div>
          <div>
            <label for=\"goalie-search\">Search goalie</label>
            <input id=\"goalie-search\" type=\"search\" placeholder=\"Type a goalie name...\" />
          </div>
          <div>
            <label>&nbsp;</label>
            <button id=\"reset-view\">Reset view</button>
          </div>
        </div>

        <div id=\"savepct-chart\"></div>

        <div class=\"stats-grid\" id=\"league-stats\"></div>
      </div>

      <div class=\"panel\">
        <div style=\"display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;\">
          <h2 style=\"margin: 0;\">Goalie summary</h2>
          <div class=\"tag\" id=\"active-league-name\"></div>
        </div>
        <table>
          <thead>
            <tr>
              <th data-key=\"goalie\" class=\"sortable\">Goalie</th>
              <th data-key=\"team\" class=\"sortable\">Team</th>
              <th data-key=\"games_played\" class=\"sortable\">Games</th>
              <th data-key=\"total_shots\" class=\"sortable\">Shots</th>
              <th data-key=\"goals_against\" class=\"sortable\">GA</th>
              <th data-key=\"avg_shots_per_game\" class=\"sortable\">Avg shots</th>
              <th data-key=\"final_save_pct\" class=\"sortable\">Save %</th>
            </tr>
          </thead>
          <tbody id=\"summary-body\"></tbody>
        </table>
        <div class=\"dropped-list\" id=\"dropped-goalies\"></div>
      </div>

      <footer>
        Built with <span aria-hidden=\"true\">🏑</span> using statistik.innebandy.se data. Cumulative save percentage is calculated from saves/shots over time; goalies with zero shots are excluded from the chart but listed above.
      </footer>
    </div>

    <script>
      const leagues = {league_json};

      function populateLeagueSelector() {
        const selector = document.getElementById('league-selector');
        selector.innerHTML = '';
        leagues.forEach((league) => {
          const option = document.createElement('option');
          option.value = league.key;
          option.textContent = league.name;
          selector.appendChild(option);
        });
      }

      function formatNumber(n) {
        return new Intl.NumberFormat('sv-SE').format(n);
      }

      function renderStats(league) {
        const stats = league.stats;
        const container = document.getElementById('league-stats');
        container.innerHTML = '';
        const entries = [
          ['Goalies', stats.total_goalies],
          ['Teams', stats.total_teams],
          ['Date range', stats.date_range],
          ['Zero-shot goalies', stats.zero_shot_goalies],
        ];
        entries.forEach(([label, value]) => {
          const card = document.createElement('div');
          card.className = 'stat-card';
          card.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>`;
          container.appendChild(card);
        });
      }

      function renderTable(records) {
        const tbody = document.getElementById('summary-body');
        tbody.innerHTML = '';
        records.forEach((row) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${row.goalie}</td>
            <td>${row.team ?? ''}</td>
            <td>${row.games_played}</td>
            <td>${formatNumber(row.total_shots)}</td>
            <td>${formatNumber(row.goals_against)}</td>
            <td>${row.avg_shots_per_game.toFixed(1)}</td>
            <td>${row.save_pct_display}</td>
          `;
          tbody.appendChild(tr);
        });
      }

      function renderDropped(list) {
        const target = document.getElementById('dropped-goalies');
        if (!list || list.length === 0) {
          target.textContent = '';
          return;
        }
        target.innerHTML = `<div class="muted">Filtered out (0 shots): ${list.join(', ')}</div>`;
      }

      function sortBy(key, ascending) {
        return (a, b) => {
          const av = a[key];
          const bv = b[key];
          if (av === bv) return 0;
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          const comp = av > bv ? 1 : -1;
          return ascending ? comp : -comp;
        };
      }

      function applyFilters(records) {
        const team = document.getElementById('team-filter').value;
        const query = document.getElementById('goalie-search').value.trim().toLowerCase();
        return records.filter((row) => {
          const matchesTeam = team === 'ALL' || row.team === team;
          const matchesQuery = !query || row.goalie.toLowerCase().includes(query);
          return matchesTeam && matchesQuery;
        });
      }

      let summaryData = [];
      let sortState = { key: 'team', ascending: true };
      let traceGoalies = [];

      function restyleChart() {
        if (traceGoalies.length === 0) return;
        const team = document.getElementById('team-filter').value;
        const query = document.getElementById('goalie-search').value.trim().toLowerCase();
        const visibility = traceGoalies.map((goalie) => {
          const matchesTeam = team === 'ALL' || (goalie in currentLeague.goalie_to_team && currentLeague.goalie_to_team[goalie] === team);
          const matchesQuery = !query || goalie.toLowerCase().includes(query);
          return matchesTeam && matchesQuery;
        });
        Plotly.restyle('savepct-chart', { visible: visibility }, Array.from(traceGoalies.keys()));
      }

      function loadLeague(key) {
        currentLeague = leagues.find((league) => league.key === key) ?? leagues[0];
        summaryData = currentLeague.summary_records;
        traceGoalies = currentLeague.trace_goalies;
        document.getElementById('active-league-name').textContent = currentLeague.name;

        const teamSelect = document.getElementById('team-filter');
        teamSelect.innerHTML = '<option value="ALL">All teams</option>' + currentLeague.team_options.map((team) => `<option value="${team}">${team}</option>`).join('');

        const sorted = summaryData.sort(sortBy(sortState.key, sortState.ascending));
        renderTable(applyFilters(sorted));
        renderDropped(currentLeague.dropped_goalies);
        renderStats(currentLeague);

        Plotly.newPlot('savepct-chart', currentLeague.figure.data, currentLeague.figure.layout, { responsive: true });
        restyleChart();
      }

      let currentLeague = null;

      window.addEventListener('DOMContentLoaded', () => {
        populateLeagueSelector();
        loadLeague(leagues[0].key);

        document.getElementById('league-selector').addEventListener('change', (event) => {
          loadLeague(event.target.value);
        });

        document.getElementById('team-filter').addEventListener('change', () => {
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        });

        document.getElementById('goalie-search').addEventListener('input', () => {
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        });

        document.getElementById('reset-view').addEventListener('click', () => {
          sortState = { key: 'team', ascending: true };
          document.getElementById('team-filter').value = 'ALL';
          document.getElementById('goalie-search').value = '';
          renderTable(summaryData.sort(sortBy(sortState.key, sortState.ascending)));
          traceGoalies.forEach((goalie, index) => {
            Plotly.restyle('savepct-chart', { visible: true }, [index]);
          });
        });

        document.querySelectorAll('th.sortable').forEach((th) => {
          th.addEventListener('click', () => {
            const key = th.dataset.key;
            if (sortState.key === key) {
              sortState.ascending = !sortState.ascending;
            } else {
              sortState = { key, ascending: true };
            }
            const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
            renderTable(filtered);
          });
        });

        Plotly.newPlot('savepct-chart', leagues[0].figure.data, leagues[0].figure.layout, { responsive: true });
        Plotly.d3.select('#savepct-chart').on('plotly_doubleclick', () => {
          document.getElementById('reset-view').click();
        });
      });
    </script>
  </body>
</html>"""


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
