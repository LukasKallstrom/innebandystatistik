# Innebandy goalie statistics scraper

Python tooling for downloading goalie statistics from `statistik.innebandy.se`,
normalising the data, and publishing both static exports and an interactive
Plotly dashboard.

## Overview
- Scrape fixture pages and extract per-goalie stats with `fetch_data.py`.
- Aggregate and visualise timelines in `display_data.py`.
- Export the results to Excel (`goalie_stats.xlsx`), a PNG timeline
  (`goalie_savepct_timeline.png` when matplotlib is available), and a standalone
  Plotly dashboard (`index.html` built from `dashboard_template.html`).

## Prerequisites
- Python 3.10 or later
- Dependencies: `requests`, `pandas`, `beautifulsoup4`, `lxml`, `plotly` (for
  the dashboard), `matplotlib` (for the PNG)

Install with pip:

```bash
python -m pip install -r requirements.txt
```

If you only need scraping/parsing, omit `plotly` and `matplotlib`.

## Run the pipeline
Run the end-to-end build to fetch the preconfigured leagues, write exports, and
render the dashboard HTML:

```bash
python display_data.py
```

The script uses hard-coded fixture URLs and takes no arguments. Network access
is required when fetching live data.

## Configure data sources
- In `fetch_data.py`, adjust `PRECONFIGURED_FIXTURE_URLS` (and related
  constants) to scrape different fixture lists.
- In `display_data.py`, update `LEAGUE_SOURCES` to change which leagues appear
  in the dashboard (mapping display names to fixture URLs).

After editing those lists, rerun `python display_data.py` to regenerate all
outputs.

## Programmatic use
Import the scraping and aggregation helpers to integrate the workflow elsewhere:

```python
from fetch_data import scrape_data, prepare_cumulative_save_percentages, write_excel

fixture_url = "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40693"
games_df, appearances_df = scrape_data(fixture_url)
cumulative, dropped_goalies = prepare_cumulative_save_percentages(games_df, appearances_df)
write_excel(games_df, appearances_df, "goalie_stats.xlsx")
```

Reuse `display_data.build_dashboard_html` if you want to embed the interactive
view in a different HTML shell.

## Testing
Smoke tests for the HTML parsing live in `test_main.py`. Run them with:

```bash
python -m pytest
```
