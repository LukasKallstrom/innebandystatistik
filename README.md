# Innebandy Goalie Stats Toolkit

This project provides a minimal, dependency-light toolkit for parsing goalie statistics
from HTML tables and aggregating them into reusable JSON. The previous ad-hoc scripts
have been replaced with a small, well-tested library and command line interface.

## Components
- **`innebandy.parser`** – resilient HTML parser that tolerates missing columns,
  infers shots against when necessary, and validates basic numerical consistency.
- **`innebandy.aggregator`** – aggregation helpers that collapse multiple appearances
  into per-goalie summaries.
- **`main.py`** – CLI wrapper that reads a JSON config describing games and renders a
  combined JSON payload with both appearances and aggregated summaries.

## Requirements
- Python 3.10+

## Usage
Prepare a JSON configuration file describing the games you want to parse. Each entry
requires an `id` and either an inline `html` fragment or the name of an HTML file
(`html_path`) relative to the config file.

```json
[
  {"id": "G1", "home": "Falcons", "away": "Wolves", "html_path": "g1.html"},
  {"id": "G2", "home": "Wolves", "away": "Falcons", "html_path": "g2.html"}
]
```

Run the CLI to parse all games and write a JSON summary:

```bash
python main.py --config games.json --output stats.json
```

The output contains two arrays: `appearances` (one entry per goalie per game) and
`summaries` (totals per goalie/team combination including save percentage and time on
ice).

## Testing
The repository is covered by a comprehensive pytest suite, including CLI execution
checks. Run it with:

```bash
python -m pytest
```
