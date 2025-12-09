# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational project building a profanity filter for gaming chat, based on real-world production work from Singularity 6's game Palia.

**Critical: The four levels are NOT sequential steps - they are parallel, alternative approaches to solving the same problem.** Each level represents a complete, standalone solution using different techniques. The goal is to implement multiple approaches and compare their tradeoffs (accuracy, speed, cost, complexity, maintainability).

### Four Alternative Approaches

1. **Level 1 - Rule-Based**: Complete solution using regex patterns and word lists
2. **Level 2 - LLM-Based**: Complete solution using LLM APIs (OpenRouter)
3. **Level 3 - Traditional ML**: Complete solution using scikit-learn classifiers
4. **Level 4 - Advanced ML**: Complete solutions using transformers, multilingual models, or deployment packaging

**Each approach solves the full problem independently.** You'll evaluate and compare them to understand when each excels.

## Project Structure

Each level is a separate Python project with its own virtual environment, initialized using `uv init --venv`:

```
tek_innov8rs_profanity_filter/
├── level1-rule-based/       # Level 1: Rule-based approach
│   ├── .venv/              # Isolated virtual environment
│   ├── pyproject.toml      # Dependencies for this level
│   ├── src/                # Source code
│   └── ...
├── level2-llm-based/        # Level 2: LLM-based approach
│   ├── .venv/
│   ├── pyproject.toml
│   ├── src/
│   └── ...
├── level3-traditional-ml/   # Level 3: Traditional ML approach
│   ├── .venv/
│   ├── pyproject.toml
│   ├── src/
│   └── ...
├── level4-advanced/         # Level 4: Advanced approaches
│   ├── .venv/
│   ├── pyproject.toml
│   └── ...
├── data/                    # Shared datasets (gitignored)
│   ├── GameTox/            # Gaming chat with toxicity labels
│   └── reddit-usernames/   # Unlabeled usernames for testing
├── docs/                    # Documentation and research
│   └── Perplexity_note.md  # Datasets, libraries, and challenges catalog
├── .claude/                 # Custom Claude Code configuration
│   ├── agents/             # Custom agents (@research-documentation-agent)
│   └── commands/           # Slash commands (/idea-next, /fix-issue, etc.)
├── CLAUDE.md               # This file
└── README.md               # Detailed level-by-level instructions
```

### Setting Up a New Level

```bash
# Create the directory
mkdir level1-rule-based

# Initialize with uv (creates pyproject.toml, .venv/, src/ structure)
cd level1-rule-based
uv init --venv

# Add dependencies for this level
uv add <package-name>

# Reference shared data from parent directory
# Use ../data/ in your code
```

Each level is self-contained with isolated dependencies, but all share the `data/` directory at the root.

## Key Datasets

Both datasets must be manually downloaded to `data/`:

- **GameTox**: Gaming chat messages with labels (clean/profanity/insult/hate speech). Download from https://github.com/shucoll/GameTox
- **Reddit Usernames**: Real usernames for testing false positives. Download from https://www.kaggle.com/datasets/colinmorris/reddit-usernames

The `data/` directory is gitignored to avoid committing large files.

## Environment Setup

**Package Management**: This project uses **`uv`** instead of `pip` for all package management, dependency installation, and virtual environment operations.

- Install packages: `uv pip install <package>`
- Create virtual environment: `uv venv`
- Activate environment: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
- Install from requirements: `uv pip install -r requirements.txt`
- **Never use `pip` directly - always use `uv`**

**API Keys**: Store in `.env` file (gitignored):
```
OPENROUTER_API_KEY=your_key_here
```

Required for Level 2 (LLM-based filter). Get free key from https://openrouter.ai/

**Python Dependencies**: This project is CLI/notebook-based. Common dependencies:
- `scikit-learn` - For Level 3 ML classifiers
- `better-profanity` or `alt-profanity-check` - Baseline comparisons
- `requests` - For OpenRouter API calls
- `python-dotenv` - For environment variable management
- `jupyter` or `marimo` - For notebook exploration (optional)

Install as needed based on the level being implemented using `uv pip install <package>`.

## Development Workflow

### Custom Slash Commands

This project uses custom slash commands defined in `.claude/commands/`:

- `/idea-next` - Read `ideas.md` and select next idea to work on, create feature branch
- `/idea-create` - Add new ideas to `ideas.md`
- `/idea-done` - Mark current idea as complete
- `/fix-issue` - Fix a GitHub issue
- `/create-issue` - Create a GitHub issue
- `/merge-pr` - Merge a pull request

### Research Agent

Use `@research-documentation-agent` when researching:
- New ML techniques or algorithms
- Profanity filtering approaches
- Dataset documentation
- Library comparisons

The agent automatically creates markdown documentation in `docs/research/` following the style guide in `.claude/DOCUMENTATION_STYLE_GUIDE.md`.

## Real-World Context

This project simulates production constraints from Palia:

- **Scale**: ~1M messages/day across 10+ languages
- **Platform requirements**: Must satisfy Microsoft XR-018, Nintendo, and Sony certification
- **UX considerations**: Clear feedback for false positives (e.g., "cass" blocked due to "ass")
- **Adversarial users**: Active attempts to bypass filters with leetspeak, Unicode, spacing

## Evaluation Metrics

Each approach should be evaluated on the SAME test data for fair comparison:

- **Precision**: Of flagged messages, what % are actually toxic?
- **Recall**: Of toxic messages, what % did we catch?
- **F1-score**: Harmonic mean of precision and recall (preferred for imbalanced data)
- **Accuracy**: Overall correctness (less useful with rare positive class)
- **Latency**: Time to classify a single message (production constraint)
- **Cost**: API costs per message (for LLM approach) or development/maintenance effort
- **Complexity**: Implementation difficulty, dependencies, deployment requirements

Binary classification: profane vs. clean (GameTox has multiple categories but start with binary).

**Key insight**: No single approach wins on all dimensions. Rule-based may be fastest but least accurate; LLMs most accurate but expensive; traditional ML is the middle ground.

## Common Pitfalls

1. **False positives**: Legitimate words containing profane substrings (Scunthorpe problem: "assassin", "mishit", "cassette")
2. **Context matters**: "That's sick!" (positive) vs "You're sick" (negative)
3. **Leetspeak variants**: h3ll, sh1t, f*ck, a$$
4. **Cross-language profanity**: English curse words in non-English contexts
5. **Dataset biases**: GameTox is English-only, from one game (World of Tanks)

## Implementation Notes

**Important: Each level is a different, complete approach to the problem.** They can be implemented in any order. The goal is to build multiple solutions and compare their tradeoffs.

**Level 1 - Rule-Based**:
- Start with 5-10 words, then expand
- Use `\b` word boundaries in regex to reduce false positives
- Compare against `alt-profanity-check` as baseline
- Expect precision/recall tradeoff

**Level 2 - LLM-Based**:
- Use OpenRouter for free model access: `openai/gpt-oss-20b:free`, `meta-llama/llama-3.3-70b-instruct:free`
- Calculate costs for paid models to assess production feasibility
- Prompt engineering significantly impacts accuracy
- Consider structured/JSON output mode for reliable parsing

**Level 3 - Traditional ML**:
- Split GameTox into train/test sets (don't use test data during training!)
- Start with `TfidfVectorizer(ngram_range=(1, 1))` and `LogisticRegression`
- Use scikit-learn `Pipeline` for clean deployment
- Experiment with hyperparameters: `ngram_range=(1, 2)`, `min_df`, `C` values

**Level 4 - Advanced**:
- ModernBERT or toxic-bert for transformer approaches
- Multilingual: Validate data quality carefully for languages you don't speak
- Additional datasets listed in `docs/Perplexity_note.md`

## Critical References

- **Production requirements**: Microsoft XR-018 at https://learn.microsoft.com/en-us/gaming/gdk/docs/store/policies/xr/xr018
- **Datasets catalog**: `docs/Perplexity_note.md` - comprehensive list of labeled datasets, Python libraries, and research
- **Level instructions**: `README.md` - detailed tasks and definitions for each level
- **Documentation style**: `.claude/DOCUMENTATION_STYLE_GUIDE.md` - follow for any research documentation

## Output Expectations

- CLI scripts preferred over notebooks (notebooks acceptable for exploration)
- All code in Python
- Clear evaluation reports with precision/recall/F1
- Analysis of false positives and false negatives
- Comparison across approaches (rule-based vs LLM vs ML)
- When we have just enough tokens left to write a summary file for the session. Please inform the situation and do so. THis way I dont have to wait for token resets and I can just start a new session and ask you to read the summary.