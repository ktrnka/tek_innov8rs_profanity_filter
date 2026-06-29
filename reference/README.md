# Reference Implementation

> ⚠️ **Spoilers — for students: try each level yourself first.** This folder is the instructor's
> reference implementation of the profanity filter described in the top-level
> [`README.md`](../README.md). The whole point of the project is to build it yourself; reading the
> finished code before you've struggled with a level short-circuits the learning. Come back here to
> compare approaches *after* you've made your own attempt.

This is a self-contained Python project (managed with [uv](https://docs.astral.sh/uv/)). Run all
commands **from this `reference/` directory** — the code resolves data paths relative to the current
working directory (e.g. `data/GameTox/train.csv`).

## Setup

```bash
cd reference

# 1. Get the data (GameTox is required; Reddit usernames are optional, Level 4)
cd data && bash download.sh && cd ..

# 2. (LLM level only) add a Gemini API key
cp .env.example .env   # then edit .env and set GEMINI_API_KEY=...
```

`data/` and `models/` are gitignored, so the dataset and any trained models stay local.

## Running the filters

The CLI is grouped by level (see `main.py`):

```bash
uv run main.py regex   --help    # Level 1: rule-based
uv run main.py sklearn --help    # Level 2: TF-IDF + LogisticRegression
uv run main.py llm     --help    # Level 3: LLM filter
uv run main.py gemini  --help    # Level 3: Gemini-specific

# examples
uv run main.py sklearn train                 # train + evaluate on a held-out split
uv run main.py sklearn train --char-ngrams   # character n-gram variant (Level 4)
uv run main.py sklearn predict "you noob"
```

## Docs

- [`docs/2026-06-reference-results.md`](docs/2026-06-reference-results.md) — the instructor's running
  log of measured baselines and lessons learned (the "log every run" habit the course asks for).
- [`COURSE_REDESIGN_2026.md`](COURSE_REDESIGN_2026.md) — planning notes for the June 2026 redesign.

Student-facing notes (Q&A, dataset options) live in the top-level [`docs/`](../docs/) folder.
