# Innebandy Goalie Statistics Toolkit

## Overview
This repository contains a small toolkit for collecting and visualising Swedish floorball (innebandy) goalie statistics. The core scraper downloads match data from [statistik.innebandy.se](http://statistik.innebandy.se), normalises the goalie appearances into tidy data frames, and exports a ready-to-use Excel workbook. An optional companion script transforms the workbook (or freshly scraped data) into an interactive Plotly dashboard that tracks each goalie's cumulative save percentage throughout the season.

The project favours portability: it is implemented in pure Python, ships without obscure runtime dependencies, and can be executed on macOS, Linux, or Windows as long as Python 3.10+ is available.

## Features
- **Fixture scraper** – Traverses a season fixture list, follows every match link, and extracts goalie statistics including saves, shots against, goals against, save percentage, and time on ice when available.
- **Clean Excel output** – Produces an `goalie_stats.xlsx` workbook with separate sheets for match metadata and goalie appearances, making further analysis or visualisation straightforward.
- **Static PNG chart** – Optionally renders a Matplotlib save-percentage timeline (one trace per goalie) for quick offline inspection.
- **Interactive dashboard** – Generates an HTML/Plotly experience with filtering, search, cumulative trends, and sortable summary tables.
- **Robust parsing** – Heuristics cope with multiple markup variants that the statistik.innebandy.se website has used across seasons.
- **Debug utilities** – Optional CSV exports help diagnose parsing issues when site markup changes.

## Repository structure
```
.
├── assets/                     # Static assets referenced by the dashboard
├── main.py                     # Scraper CLI entry point
├── interactive_savepct.py      # Interactive Plotly dashboard generator
├── goalie_stats.xlsx           # Example workbook produced by the scraper
├── index.html / teams.html ... # Example rendered dashboards
├── test_main.py                # Unit tests covering the parser heuristics
└── goalie_savepct_timeline.png # Example Matplotlib output
```

## Requirements
- Python **3.10 or later**
- Python packages:
  - Required: [`requests`](https://pypi.org/project/requests/), [`pandas`](https://pypi.org/project/pandas/), [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/), [`lxml`](https://pypi.org/project/lxml/)
  - Optional (for extra features):
    - [`matplotlib`](https://pypi.org/project/matplotlib/) for the static save-percentage PNG
    - [`plotly`](https://pypi.org/project/plotly/) for the interactive dashboard

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install --upgrade pip
pip install pandas requests beautifulsoup4 lxml matplotlib plotly
```

## Usage
### 1. Scrape goalie statistics and build the Excel workbook
```bash
python main.py --season-url "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701" --output goalie_stats.xlsx --plot goalie_savepct_timeline.png
```
Key options:
- `--season-url` – Fixture list to process. Defaults to the Damallsvenskan example included above.
- `--output` – Destination Excel workbook. Defaults to `goalie_stats.xlsx` in the current directory.
- `--plot` – Optional Matplotlib PNG path. If Matplotlib is not installed the script silently skips plot generation.
- `--verbose` / `--quiet` – Adjust console logging verbosity.

> **Tip:** When the website markup changes, re-run with `--verbose` to capture detailed logs. Combined with the `--debug-csv` flag in `interactive_savepct.py` you can persist raw tables for closer inspection when adapting the parser.

### 2. Build the interactive cumulative save-percentage dashboard
Use either the workbook created above or ask the script to scrape data on the fly:

```bash
# Using an existing workbook
python interactive_savepct.py --excel goalie_stats.xlsx --output dashboards/savepct.html

# Scrape fresh data (requires network access) and write to index.html
python interactive_savepct.py --season-url "http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701"
```
Important switches:
- `--excel` – Path to a workbook produced by `main.py`. When omitted the script scrapes the fixtures specified by `--season-url`.
- `--output` – Destination HTML file (defaults to `index.html`).
- `--verbose` – Enable detailed logging.
- `--debug-csv DIR` – Write raw `games_*.csv` and `appearances_*.csv` snapshots for inspection.

Open the generated HTML file in your browser to explore the timeline, filter goalies by team, and compare summary statistics.

## Running the test suite
The repository includes a small parser-focused test suite. Install the required dependencies (including BeautifulSoup) and run:

```bash
python -m pytest
```

If BeautifulSoup (from `beautifulsoup4`) is missing, pytest will abort during collection—install the package to resolve the error.

## Troubleshooting
- **No games detected:** Ensure the `--season-url` points to a valid fixture list. Some competitions require authentication and cannot be scraped anonymously.
- **Goalie tables missing columns:** When site markup changes, re-run with `--verbose` and review the stored HTML (see script docstrings) to adapt the selectors.
- **Plotly dashboard missing traces:** Ensure the workbook contains at least one goalie appearance with save-percentage data; otherwise the cumulative timeline cannot be produced.

## License
No explicit license file is bundled with this repository. Add your preferred license before distributing or open-sourcing derivative work.

