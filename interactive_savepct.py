"""Generate an interactive goalie save-percentage timeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

try:
    import plotly.express as px
except ImportError as exc:
    raise SystemExit(
        "plotly is required for the interactive visualisation. Install it via 'pip install plotly'."
    ) from exc

try:
    from main import (
        Appearance,
        Game,
        DEFAULT_FIXTURE_URL,
        appearances_to_frame,
        build_session,
        fetch_html,
        games_to_frame,
        iter_game_links,
        parse_game,
    )
except ImportError as exc:
    raise SystemExit("This script must be executed from the project root.") from exc


# -----------------------
# I/O helpers
# -----------------------

def _write_debug_csv(dirpath: Path, games_df: pd.DataFrame, apps_df: pd.DataFrame, tag: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    gpath = dirpath / f"games_{tag}.csv"
    apath = dirpath / f"appearances_{tag}.csv"
    games_df.to_csv(gpath, index=False)
    apps_df.to_csv(apath, index=False)
    logger.info("Wrote debug CSV: %s", gpath.resolve())
    logger.info("Wrote debug CSV: %s", apath.resolve())


def _diag_series(df: pd.DataFrame, cols: List[str]) -> str:
    parts = []
    total = len(df)
    parts.append(f"rows={total}")
    for c in cols:
        if c in df.columns:
            nn = int(df[c].notna().sum())
            parts.append(f"{c}.notna={nn}")
        else:
            parts.append(f"{c}=MISSING")
    return ", ".join(parts)


# -----------------------
# Scraping
# -----------------------

def scrape_data(season_url: str, debug_csv_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape *season_url* and return dataframes for games and goalie appearances."""
    session = build_session()
    logger.info("Fetching fixture list: %s", season_url)
    fixture_doc = fetch_html(session, season_url)
    game_links = sorted(set(iter_game_links(fixture_doc, season_url)))
    logger.info("Discovered %d unique games", len(game_links))

    games: list[Game] = []
    appearances: list[Appearance] = []

    for index, url in enumerate(game_links, start=1):
        logger.debug("Fetching game %d/%d: %s", index, len(game_links), url)
        try:
            game_doc = fetch_html(session, url)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue
        game, game_apps = parse_game(game_doc, url)
        games.append(game)
        appearances.extend(game_apps)
        if not game_apps:
            logger.debug("No goalie appearances parsed for %s", url)

    if not games:
        raise SystemExit("No games detected. Check that the fixture URL is correct.")

    games_df = games_to_frame(games)
    apps_df = appearances_to_frame(appearances)

    if debug_csv_dir is not None:
        _write_debug_csv(debug_csv_dir, games_df, apps_df, tag="raw")

    # Snabb diagnostik
    logger.info("Parsed goalie appearances: %d rows", len(apps_df))
    logger.info("Appearances diagnostic: %s", _diag_series(apps_df, ["goalie", "save_pct", "saves", "shots_against", "goals_against"]))

    return games_df, apps_df


# -----------------------
# Load from Excel
# -----------------------

def load_from_excel(path: Path, debug_csv_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load games and appearances from an Excel workbook created by main.py."""
    logger.info("Loading data from %s", path)
    with pd.ExcelFile(path) as workbook:
        games = pd.read_excel(workbook, "games")
        appearances = pd.read_excel(workbook, "appearances")

    if debug_csv_dir is not None:
        _write_debug_csv(debug_csv_dir, games, appearances, tag="from_excel")

    logger.info("Appearances diagnostic: %s", _diag_series(appearances, ["goalie", "save_pct", "saves", "shots_against", "goals_against"]))
    return games, appearances


# -----------------------
# Prep
# -----------------------

def _to_pct01(series: pd.Series) -> pd.Series:
    """Parse save_pct values to [0..1] floats from strings like '78,6 %', '78%', '0.786'."""
    s = series.astype(str).str.strip()
    s = (
        s.str.replace("\u202f", "", regex=False)  # narrow no-break space
         .str.replace("\xa0", "", regex=False)    # NBSP
         .str.replace("%", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    s = s.mask(s.eq("") | s.eq("nan"))
    out = pd.to_numeric(s, errors="coerce")
    need_div = out > 1.5
    out.loc[need_div] = out.loc[need_div] / 100.0
    return out


def prepare_cumulative_save_percentages(
    games: pd.DataFrame, appearances: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (cumulative_df, dropped_goalies) with cumulative save% per goalie over time.
    Robust mot saknade datum: faller tillbaka till syntetiska datum baserat på matchordning.
    """

    if games.empty:
        raise ValueError("Games dataframe is empty")
    if appearances.empty:
        raise ValueError(
            "No goalie appearances found (0 rows). "
            "If you scraped just now, it likely means the site markup didn't expose goalie stats. "
            "Try running main.py with '--debug-html DIR' to inspect a match page."
        )

    required_columns = {"game_id", "goalie"}
    missing_required = required_columns.difference(appearances.columns)
    if missing_required:
        raise ValueError(f"Missing required appearance columns: {', '.join(sorted(missing_required))}")

    # Säkerställ kolumner
    for col in ("saves", "shots_against", "goals_against", "save_pct"):
        if col not in appearances.columns:
            appearances[col] = pd.NA

    # Lägg till en stabil ordning per game_id (matchernas ordning i 'games')
    games = games.copy()
    games["__order"] = range(len(games))

    # Merge in date + order
    merged = appearances.merge(
        games[["game_id", "date", "__order"]],
        on="game_id",
        how="left",
        validate="many_to_one",
    )

    # Numerik & procent
    for col in ("saves", "shots_against", "goals_against"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    def _to_pct01(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = (s.str.replace("\u202f", "", regex=False)
               .str.replace("\xa0", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.replace(",", ".", regex=False))
        s = s.mask(s.eq("") | s.eq("nan"))
        out = pd.to_numeric(s, errors="coerce")
        out.loc[out > 1.5] = out.loc[out > 1.5] / 100.0
        return out

    merged["save_pct"] = _to_pct01(merged["save_pct"])

    # --- Rekonstruktion ---
    m = merged  # alias

    # 1) shots = saves + goals
    mask = m["shots_against"].isna() & m["saves"].notna() & m["goals_against"].notna()
    m.loc[mask, "shots_against"] = m.loc[mask, "saves"] + m.loc[mask, "goals_against"]

    # 2) saves = shots - goals
    mask = m["saves"].isna() & m["shots_against"].notna() & m["goals_against"].notna()
    m.loc[mask, "saves"] = m.loc[mask, "shots_against"] - m.loc[mask, "goals_against"]

    # 3) saves from save_pct & shots
    mask = m["saves"].isna() & m["shots_against"].notna() & m["save_pct"].notna()
    m.loc[mask, "saves"] = (m.loc[mask, "save_pct"] * m.loc[mask, "shots_against"]).round().astype("Int64")

    # 4) shots & saves from save_pct & goals
    mask = m["shots_against"].isna() & m["goals_against"].notna() & m["save_pct"].notna()
    denom = 1.0 - m.loc[mask, "save_pct"]
    feasible = denom > 1e-9
    idxs = denom[feasible].index
    est_shots = (m.loc[idxs, "goals_against"] / denom.loc[idxs]).round()
    m.loc[idxs, "shots_against"] = est_shots
    m.loc[idxs, "saves"] = (m.loc[idxs, "shots_against"] - m.loc[idxs, "goals_against"]).astype("Int64")

    # 5) goals from shots & saves
    mask = m["goals_against"].isna() & m["shots_against"].notna() & m["saves"].notna()
    m.loc[mask, "goals_against"] = (m.loc[mask, "shots_against"] - m.loc[mask, "saves"]).astype("Int64")

    # Om datum saknas → skapa syntetiska datum utifrån __order
    # (Använd en fast bas så att timeline blir stabil.)
    missing_dates = m["date"].isna()
    if missing_dates.any():
        base = pd.Timestamp("2000-01-01")
        # Fyll ev. saknad __order (om merge misslyckades för någon rad)
        m["__order"] = pd.to_numeric(m["__order"], errors="coerce")
        m["__order"] = m["__order"].fillna(m["__order"].min(skipna=True)).fillna(0).astype(int)
        m.loc[missing_dates, "date"] = base + pd.to_timedelta(m.loc[missing_dates, "__order"], unit="D")
        logger.warning(
            "Dates missing for %d appearances — using synthetic dates based on match order.",
            int(missing_dates.sum()),
        )

    # Kräver minst goalie & någon siffra (zeros OK, NaN ej)
    m = m.dropna(subset=["date", "goalie"])
    m = m[~(m["saves"].isna() & m["shots_against"].isna())]

    # En sista fyllning om det går
    mask = m["saves"].isna() & m["shots_against"].notna() & m["goals_against"].notna()
    m.loc[mask, "saves"] = m.loc[mask, "shots_against"] - m.loc[mask, "goals_against"]
    mask = m["shots_against"].isna() & m["saves"].notna() & m["goals_against"].notna()
    m.loc[mask, "shots_against"] = m.loc[mask, "saves"] + m.loc[mask, "goals_against"]

    # Slutligen måste 'saves' och 'shots_against' finnas (0 OK)
    m = m.dropna(subset=["saves", "shots_against"])
    if m.empty:
        raise ValueError(
            "Insufficient goalie data with saves and shots even after reconstruction and synthetic dates. "
            "This suggests scraping returned no per-goalie stats."
        )

    # Sortera kronologiskt (syntetiska datum respekterar matchordning)
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values(["date", "game_id"])

    # Kumulativt per målvakt
    m["saves"] = pd.to_numeric(m["saves"], errors="coerce").fillna(0).astype(int)
    m["shots_against"] = pd.to_numeric(m["shots_against"], errors="coerce").fillna(0).astype(int)
    m["cumulative_saves"] = m.groupby("goalie")["saves"].cumsum()
    m["cumulative_shots"] = m.groupby("goalie")["shots_against"].cumsum()

    # Kumulativ rädd%: NaN tills skott > 0
    m["cumulative_save_pct"] = pd.NA
    pos_mask = m["cumulative_shots"] > 0
    m.loc[pos_mask, "cumulative_save_pct"] = (
        m.loc[pos_mask, "cumulative_saves"] / m.loc[pos_mask, "cumulative_shots"]
    )

    # Släng målvakter med 0 skott totalt (listan visas i panelen)
    shot_sum_by_goalie = m.groupby("goalie")["shots_against"].transform("sum")
    has_any_shots = shot_sum_by_goalie > 0
    dropped_goalies = m.loc[~has_any_shots, "goalie"].drop_duplicates().sort_values().tolist()
    m = m[has_any_shots]

    if m.empty:
        raise ValueError(
            "All goalies in the dataset have zero shots against; nothing to plot. "
            "This can happen if the dataset only contains 0–0 backup appearances."
        )

    if "team" in m:
        m["team"] = m["team"].fillna("Unknown")

    return m, dropped_goalies



# -----------------------
# Plot
# -----------------------

def build_figure(cumulative: pd.DataFrame, dropped_goalies: List[str]):
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
        color_discrete_sequence=px.colors.qualitative.Bold,
        custom_data=[
            "goalie",
            "team",
            "game_id",
            "cumulative_saves",
            "cumulative_shots",
        ],
    )

    fig.update_traces(
        mode="lines+markers",
        marker=dict(size=7, line=dict(width=0, color="rgba(15, 23, 42, 0.3)")),
        line=dict(width=2.5, shape="spline", smoothing=0.6),
        hovertemplate=(
            "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
            "Date: %{x|%Y-%m-%d}<br>"
            "Game: %{customdata[2]}<br>"
            "Shots: %{customdata[4]}<br>"
            "Saves: %{customdata[3]}<br>"
            "Save %: %{y:.1%}<extra></extra>"
        ),
    )

    fig.update_layout(
        title="Goalie cumulative save percentage over time",
        hovermode="x unified",
        margin=dict(t=80, r=24, b=80, l=60),
        legend=dict(
            title="Goalie",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.05)",
        ),
        plot_bgcolor="rgba(248, 250, 252, 0.98)",
        paper_bgcolor="rgba(15, 23, 42, 0)",
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(
            rangeslider=dict(visible=True, bgcolor="rgba(148, 163, 184, 0.25)"),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.25)",
            zeroline=False,
            linecolor="rgba(148, 163, 184, 0.5)",
        ),
        yaxis=dict(
            tickformat=".0%",
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.2)",
            linecolor="rgba(148, 163, 184, 0.5)",
        ),
        hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.85)", font=dict(color="#f8fafc")),
    )

    return fig


def summarise_goalies(cumulative: pd.DataFrame) -> pd.DataFrame:
    """Return a per-goalie summary suitable for the dashboard data table."""

    if "team" not in cumulative.columns:
        cumulative = cumulative.copy()
        cumulative["team"] = "Unknown"

    cumulative["team"] = cumulative["team"].fillna("Unknown")
    ordered = cumulative.sort_values("date")
    grouped = ordered.groupby(["team", "goalie"], dropna=False)

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

    summary = summary.sort_values(["team", "goalie"]).reset_index(drop=True)
    return summary


def build_dashboard_html(
    figure,
    cumulative: pd.DataFrame,
    summary: pd.DataFrame,
    dropped_goalies: List[str],
) -> str:
    """Compose the interactive dashboard HTML shell."""

    import plotly.io as pio

    chart_html = pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        div_id="savepct-chart",
        config={
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )

    cumulative = cumulative.copy()
    cumulative["team"] = cumulative.get("team", "Unknown").fillna("Unknown")

    goalie_team_map = (
        cumulative[["goalie", "team"]]
        .drop_duplicates()
        .sort_values("goalie")
        .assign(team=lambda df: df["team"].fillna("Unknown"))
    )
    goalie_to_team = {row.goalie: row.team for row in goalie_team_map.itertuples(index=False)}

    team_options = sorted({team for team in cumulative["team"].dropna().unique()})

    summary_records = summary.replace({pd.NA: None}).to_dict(orient="records")

    total_goalies = len({row["goalie"] for row in summary_records})
    total_teams = len(team_options)
    date_min = pd.to_datetime(cumulative["date"]).min()
    date_max = pd.to_datetime(cumulative["date"]).max()
    date_range = "–"
    if pd.notna(date_min) and pd.notna(date_max):
        date_range = f"{date_min.date().isoformat()} → {date_max.date().isoformat()}"

    dropdown_options = "".join(
        f'<option value="{team}">{team}</option>' for team in team_options
    )

    dropped_html = ""
    if dropped_goalies:
        items = "".join(f"<li>{name}</li>" for name in dropped_goalies)
        dropped_html = (
            "<div class=\"callout\" role=\"status\">"
            "  <h3>Filtered goalies (0 recorded shots)</h3>"
            f"  <ul>{items}</ul>"
            "</div>"
        )

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
        gap: 0.75rem;
      }}

      header h1 {{
        font-size: clamp(1.75rem, 4vw, 2.75rem);
        margin: 0;
        letter-spacing: -0.02em;
      }}

      header p {{
        margin: 0;
        color: rgba(226, 232, 240, 0.85);
        max-width: 60ch;
      }}

      .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
      }}

      .stat-card {{
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        backdrop-filter: blur(8px);
      }}

      .stat-card h2 {{
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.25rem;
        color: rgba(148, 163, 184, 0.85);
      }}

      .stat-card span {{
        font-size: 1.8rem;
        font-weight: 600;
      }}

      .panel {{
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.72), rgba(30, 41, 59, 0.72));
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 18px;
        padding: 1.5rem;
        box-shadow: 0 25px 45px rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
      }}

      .panel h2 {{
        margin-top: 0;
        font-size: 1.3rem;
        margin-bottom: 1rem;
      }}

      .controls {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
      }}

      .controls label {{
        font-size: 0.9rem;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        color: rgba(226, 232, 240, 0.9);
      }}

      .controls select,
      .controls input {{
        min-width: 240px;
        padding: 0.6rem 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background-color: rgba(248, 250, 252, 0.1);
        color: white;
        font-size: 0.95rem;
      }}

      .controls button {{
        align-self: flex-end;
        padding: 0.6rem 1.1rem;
        border-radius: 12px;
        border: none;
        background: linear-gradient(120deg, rgba(59, 130, 246, 0.9), rgba(99, 102, 241, 0.85));
        color: white;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }}

      .controls button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 15px 30px rgba(59, 130, 246, 0.35);
      }}

      #savepct-chart {{
        width: 100%;
        min-height: 480px;
        border-radius: 16px;
        overflow: hidden;
        background: rgba(248, 250, 252, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        font-size: 0.95rem;
      }}

      thead {{
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        color: rgba(148, 163, 184, 0.8);
      }}

      th, td {{
        padding: 0.65rem 0.5rem;
        text-align: left;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }}

      tbody tr:hover {{
        background-color: rgba(148, 163, 184, 0.12);
      }}

      th.sortable {{
        cursor: pointer;
      }}

      th.sortable::after {{
        content: '\\25B4\\25BE';
        margin-left: 0.35rem;
        font-size: 0.75rem;
        opacity: 0.45;
      }}

      .empty-state {{
        margin: 1.5rem 0 0;
        color: rgba(226, 232, 240, 0.7);
      }}

      .callout {{
        margin: 1.75rem 0 0;
        padding: 1.1rem 1.35rem;
        border-radius: 16px;
        border: 1px solid rgba(59, 130, 246, 0.45);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(96, 165, 250, 0.15));
        color: rgba(226, 232, 240, 0.95);
        box-shadow: inset 0 1px 0 rgba(148, 197, 255, 0.2);
      }}

      .callout h3 {{
        margin: 0 0 0.75rem;
        font-size: 1rem;
        letter-spacing: 0.02em;
      }}

      .callout ul {{
        margin: 0;
        padding-left: 1.2rem;
        display: grid;
        gap: 0.35rem;
      }}

      .callout li {{
        margin: 0;
      }}

      @media (max-width: 768px) {{
        .controls label {{
          width: 100%;
        }}
        .controls button {{
          width: 100%;
        }}
      }}
    </style>
  </head>
  <body>
    <div class=\"page\">
      <header>
        <h1>Goalie Save Percentage Dashboard</h1>
        <p>Explore how every goalie performs throughout the season. Filter by team, search for a specific goalie, and inspect a sortable table with the latest cumulative numbers.</p>
      </header>

      <section class=\"stats-grid\">
        <article class=\"stat-card\">
          <h2>Teams tracked</h2>
          <span>{total_teams}</span>
        </article>
        <article class=\"stat-card\">
          <h2>Goalies tracked</h2>
          <span>{total_goalies}</span>
        </article>
        <article class=\"stat-card\">
          <h2>Match coverage</h2>
          <span>{date_range}</span>
        </article>
        <article class=\"stat-card\">
          <h2>Zero-shot goalies</h2>
          <span>{len(dropped_goalies)}</span>
        </article>
      </section>

      <section class=\"panel\">
        <h2>Season timeline</h2>
        <div class=\"controls\">
          <label>
            Team filter
            <select id=\"team-filter\">
              <option value=\"ALL\">All teams</option>
              {dropdown_options}
            </select>
          </label>
          <label>
            Highlight goalie
            <input id=\"goalie-search\" type=\"search\" placeholder=\"Search goalie name…\" />
          </label>
          <button id=\"reset-view\" type=\"button\">Reset view</button>
        </div>
        {chart_html}
        {dropped_html}
      </section>

      <section class=\"panel\">
        <h2>Sortable goalie table</h2>
        <table>
          <thead>
            <tr>
              <th class=\"sortable\" data-key=\"team\">Team</th>
              <th class=\"sortable\" data-key=\"goalie\">Goalie</th>
              <th class=\"sortable\" data-key=\"games_played\">Games</th>
              <th class=\"sortable\" data-key=\"total_shots\">Shots</th>
              <th class=\"sortable\" data-key=\"total_saves\">Saves</th>
              <th class=\"sortable\" data-key=\"goals_against\">Goals Against</th>
              <th class=\"sortable\" data-key=\"avg_shots_per_game\">Avg Shots/Game</th>
              <th class=\"sortable\" data-key=\"final_save_pct\">Cumulative Save%</th>
            </tr>
          </thead>
          <tbody id=\"summary-table-body\"></tbody>
        </table>
        <p class=\"empty-state\" id=\"table-empty\" hidden>No goalies match the current filters.</p>
      </section>
    </div>

    <script>
      const summaryData = {json.dumps(summary_records, ensure_ascii=False)};
      const goalieToTeam = {json.dumps(goalie_to_team, ensure_ascii=False)};
      const traceGoalies = {json.dumps([trace.name for trace in figure.data], ensure_ascii=False)};

      function renderTable(rows) {{
        const tbody = document.getElementById('summary-table-body');
        tbody.innerHTML = '';
        if (!rows.length) {{
          document.getElementById('table-empty').hidden = false;
          return;
        }}
        document.getElementById('table-empty').hidden = true;
        const fragment = document.createDocumentFragment();
        rows.forEach((row) => {{
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${{row.team || 'Unknown'}}</td>
            <td>${{row.goalie}}</td>
            <td>${{row.games_played}}</td>
            <td>${{row.total_shots}}</td>
            <td>${{row.total_saves}}</td>
            <td>${{row.goals_against}}</td>
            <td>${{(row.avg_shots_per_game || 0).toFixed(1)}}</td>
            <td>${{row.save_pct_display || '–'}}</td>`;
          fragment.appendChild(tr);
        }});
        tbody.appendChild(fragment);
      }}

      function currentFilters() {{
        return {{
          team: document.getElementById('team-filter').value,
          search: document.getElementById('goalie-search').value.trim().toLowerCase(),
        }};
      }}

      function applyFilters(data) {{
        const {{ team, search }} = currentFilters();
        return data.filter((row) => {{
          const matchesTeam = team === 'ALL' || (row.team || 'Unknown') === team;
          const matchesSearch = !search || row.goalie.toLowerCase().includes(search);
          return matchesTeam && matchesSearch;
        }});
      }}

      function restyleChart() {{
        const {{ team, search }} = currentFilters();
        traceGoalies.forEach((goalie, index) => {{
          const teamMatch = team === 'ALL' || goalieToTeam[goalie] === team;
          const searchMatch = !search || goalie.toLowerCase().includes(search);
          const visible = teamMatch && searchMatch;
          Plotly.restyle('savepct-chart', {{ visible: visible ? true : 'legendonly' }}, [index]);
        }});
      }}

      function sortBy(key, ascending) {{
        return (a, b) => {{
          const av = a[key] ?? (typeof a[key] === 'number' ? 0 : '');
          const bv = b[key] ?? (typeof b[key] === 'number' ? 0 : '');
          if (typeof av === 'number' && typeof bv === 'number') {{
            return ascending ? av - bv : bv - av;
          }}
          return ascending ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
        }};
      }}

      window.addEventListener('DOMContentLoaded', () => {{
        let sortState = {{ key: 'team', ascending: true }};
        renderTable(summaryData.sort(sortBy(sortState.key, sortState.ascending)));

        document.getElementById('team-filter').addEventListener('change', () => {{
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        }});

        document.getElementById('goalie-search').addEventListener('input', () => {{
          const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
          renderTable(filtered);
          restyleChart();
        }});

        document.getElementById('reset-view').addEventListener('click', () => {{
          sortState = {{ key: 'team', ascending: true }};
          document.getElementById('team-filter').value = 'ALL';
          document.getElementById('goalie-search').value = '';
          renderTable(summaryData.sort(sortBy(sortState.key, sortState.ascending)));
          traceGoalies.forEach((goalie, index) => {{
            Plotly.restyle('savepct-chart', {{ visible: true }}, [index]);
          }});
        }});

        document.querySelectorAll('th.sortable').forEach((th) => {{
          th.addEventListener('click', () => {{
            const key = th.dataset.key;
            if (sortState.key === key) {{
              sortState.ascending = !sortState.ascending;
            }} else {{
              sortState = {{ key, ascending: true }};
            }}
            const filtered = applyFilters(summaryData).sort(sortBy(sortState.key, sortState.ascending));
            renderTable(filtered);
          }});
        }});

        Plotly.d3.select('#savepct-chart').on('plotly_doubleclick', () => {{
          document.getElementById('reset-view').click();
        }});
      }});
    </script>
  </body>
</html>"""

# -----------------------
# CLI
# -----------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an interactive timeline showing how goalies' total save percentages "
            "develop throughout the season."
        )
    )
    parser.add_argument(
        "--excel",
        type=Path,
        help=(
            "Existing Excel workbook produced by main.py. If omitted, data is scraped "
            "from --season-url."
        ),
    )
    parser.add_argument(
        "--season-url",
        default=DEFAULT_FIXTURE_URL,
        help="Fixture list to scrape when no Excel workbook is supplied.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("index.html"),
        help="Destination HTML file for the interactive plot.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to follow progress.",
    )
    parser.add_argument(
        "--debug-csv",
        type=Path,
        default=None,
        help="Directory to write raw CSVs for games/appearances to aid debugging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.excel:
        if not args.excel.exists():
            raise SystemExit(f"The Excel file {args.excel} does not exist.")
        games_df, appearances_df = load_from_excel(args.excel, debug_csv_dir=args.debug_csv)
    else:
        games_df, appearances_df = scrape_data(args.season_url, debug_csv_dir=args.debug_csv)

    cumulative, dropped_goalies = prepare_cumulative_save_percentages(games_df, appearances_df)
    if cumulative.empty:
        raise SystemExit("No cumulative save-percentage data available to plot.")

    figure = build_figure(cumulative, dropped_goalies)
    summary = summarise_goalies(cumulative)
    html = build_dashboard_html(figure, cumulative, summary, dropped_goalies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    logger.info("Interactive plot written to %s", args.output.resolve())


if __name__ == "__main__":
    main()
