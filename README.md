# Innebandy Goalie Stats Toolkit

This project downloads the fixture list from [statistik.innebandy.se](http://statistik.innebandy.se/ft.aspx?scr=fixturelist&ftid=40701),
follows each game link, and parses goalie statistics tables into a simple JSON file.
The code avoids heavy dependencies and relies on the standard library wherever
possible.

## Components
- **`innebandy.scraper`** – discovers game links on the fixture page and fetches each
  game for parsing.
- **`innebandy.parser`** – resilient HTML parser that tolerates missing columns,
  infers shots against when necessary, and validates basic numerical consistency.
- **`innebandy.aggregator`** – aggregation helpers that collapse multiple appearances
  into per-goalie summaries.
- **`main.py`** – CLI wrapper that pulls the fixture list, parses all games, and
  writes combined appearance and summary JSON.

## Requirements
- Python 3.10+

## Usage
Run the CLI directly; it defaults to the fixture list for FTID 40701:

```bash
python main.py --output stats.json
```

To use a different season or a local HTML copy, pass a custom fixture URL (file URLs
work, which is handy for offline testing):

```bash
python main.py --fixture-url file:///path/to/fixtures.html --output stats.json
```

The output contains two arrays: `appearances` (one entry per goalie per game) and
`summaries` (totals per goalie/team combination including save percentage and time on
ice).

## Testing
Run the pytest suite, which covers scraper discovery, parsing, aggregation, and CLI
behaviour:

```bash
python -m pytest
```
