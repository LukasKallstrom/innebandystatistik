# Innebandy goalie statistics scraper

This project downloads floorball goalie statistics from `statistik.innebandy.se`,
normalises the results, and produces both Excel/PNG exports and an interactive
Plotly dashboard. The scraping (network + parsing) logic lives in
`fetch_data.py`, while the presentation logic lives in `display_data.py`.

## Requirements
- Python 3.10+
- pip-installable dependencies:
  - `requests`
  - `pandas`
  - `beautifulsoup4` and `lxml`
  - `plotly` (only needed for the interactive dashboard)
  - `matplotlib` (optional; used for the static save-percentage PNG)

Install them with:

```bash
python -m pip install -r requirements.txt
```

(If you only need scraping/parsing, you can skip `plotly` and
`matplotlib`.)

## How to run
1. Make sure the required packages are installed.
2. Run the display script; it fetches the preconfigured leagues, writes the
   Excel and PNG exports, and builds `index.html` with the interactive chart:

   ```bash
   python display_data.py
   ```

   The script uses hard-coded fixture URLs—no command-line arguments are needed
   (or accepted). Network access is required when fetching live data.

## Outputs
- `goalie_stats.xlsx`: Excel workbook with raw appearances and game metadata.
- `goalie_savepct_timeline.png`: Static matplotlib timeline of save percentage
  (created when matplotlib is available).
- `index.html`: Interactive Plotly dashboard comparing goalie save percentage
  over time across configured leagues.

## Customising the data sources
- `fetch_data.py` contains `PRECONFIGURED_FIXTURE_URLS` and related constants
  for scraping; edit these to point at different fixture lists.
- `display_data.py` lists leagues in `LEAGUE_SOURCES`, mapping a display name to
  a fixture URL. Update this list to add or remove leagues displayed in the
  dashboard.

Because URLs are defined in code, rerunning `python display_data.py` will use the
new sources automatically—no extra arguments are necessary.

## Programmatic use
If you want to integrate the scraper into another script, import from
`fetch_data.py`:

```python
from fetch_data import scrape_data, prepare_cumulative_save_percentages, write_excel

fixture_url = "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40693"
games_df, appearances_df = scrape_data(fixture_url)
cumulative, dropped_goalies = prepare_cumulative_save_percentages(games_df, appearances_df)
write_excel(games_df, appearances_df, "goalie_stats.xlsx")
```

You can then reuse `display_data.build_dashboard_html` if you want to embed the
interactive view elsewhere.
