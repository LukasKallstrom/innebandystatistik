"""Generate an interactive goalie save-percentage timeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, List

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
            showarrow=False,
            align="left",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            bgcolor="rgba(0,0,0,0.04)",
            text="<br>".join(text_lines),
            font=dict(size=12),
        )

    return fig


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
        default=Path("goalie_savepct_interactive.html"),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs="cdn")
    logger.info("Interactive plot written to %s", args.output.resolve())


if __name__ == "__main__":
    main()
