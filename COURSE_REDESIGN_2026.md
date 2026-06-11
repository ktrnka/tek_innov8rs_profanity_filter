# Profanity Filter Course — Redesign Plan (Summer 2026 Re-run)

> **Status:** Living planning document. Instructor-facing — **do not ship to the student-facing
> `main` branch.** Belongs on the private `kt_exploration` branch (or kept uncommitted).
>
> **How to use this doc:** Every decision below includes its *reasoning*, not just the action. The
> implementation plan may be flawed in places; when it is, the reasoning is what lets the implementer
> (human or agent) make a sensible judgment call instead of following a broken recipe. If you find a
> step that no longer makes sense, re-derive from the *intent* stated alongside it.

---

## 1. Context

This repo is teaching material for an advanced bootcamp cohort (junior–senior undergrad level,
~4 students), re-running **starting Sunday June 14, 2026** for a few weeks. It ran once last fall.

- **`main` branch** — student-facing material: the `README.md` brief (4 levels) + `docs/`.
- **`kt_exploration` branch** — Keith's **partial** reference implementations (the private answer key).

**Why this pass exists.** Improve the repo as a learning tool versus last time. The guiding lens, in
Keith's words:

> Assess each step and consider revisions to **reduce unproductive friction**, so that the struggle
> students experience is the *process of learning* — not the kind of struggle that isn't educational.

**Bootcamp philosophy (the constraint that shapes everything).** Students must **earn** their
knowledge. There *should* be struggle; the material must **not** become a recipe to follow blindly.
So we are not trying to remove difficulty — we are trying to remove the *wrong* difficulty. See §3.

**Observations from last time** (N≈4, too small to generalize, but concrete signals):
- OpenRouter free rate limits were too tight to do meaningful evaluation.
- A student struggled to look up terminology/concepts independently.
- Wide variance in available time → very uneven progress; some made little headway.
- One student "spray-and-pray'd" a long presentation instead of doing systematic research.
- A couple were confused by `uv` / the concept of a **package manager**, and by **`.env` files**.
- Emily's suggestion: focus the cohort on the first 2 of 4 levels and *demo* the 3rd.

---

## 2. Guiding principle: productive vs. unproductive friction

The single lens for every change. Concrete examples to calibrate against:

| Productive struggle (keep) | Unproductive friction (remove) |
| --- | --- |
| Looking up an unfamiliar term and learning the concept | A term with *zero foothold* — no pointer on where to even start |
| Discovering the precision/recall tradeoff by observing your own filter's errors | A broken dataset download or toolchain that blocks all work |
| Designing 5 good regex rules and feeling the limits | Confusing `uv`/package-manager/`.env` plumbing with no explanation |
| Systematically testing hypotheses and measuring | Confusing jargon collisions (e.g., classification "positive/negative" vs. *sentiment* positive/negative) |

When in doubt, ask: *is this difficulty teaching them the subject, or just taxing them on
incidental plumbing?* Keep the former; cut the latter.

---

## 3. Scope & level restructure

**Decision:** reorder so the well-documented, locally-runnable work is the **core**, and the
research-heavy / hardware-variable work is **optional**.

| New | Was | Role | Reasoning |
| --- | --- | --- | --- |
| **L1 Regex** | L1 | Core | Already strong; rules are tangible and need no data-science background. |
| **L2 sklearn** | L3 | **Core** | Runs locally on any hardware, abundant tutorials, deterministic. The most reliable place for a time-strapped student to get a *real ML result*. |
| **L3 LLM** | L2 | Demo / launchpad | Highest external-dependency risk (API keys, rate limits, refusals). Better as a guided demo than a required gate. |
| **L4** | L4 | Optional stretch | Hardware-variable or product-skill work that shouldn't block core technical skills. |

**Why swap sklearn ahead of LLM:** last time the LLM level's *external* friction (rate limits) hit
students before they'd built core ML intuition. sklearn is self-contained and locally reproducible,
so it's the safer core. The LLM level becomes where they *appreciate* the tradeoffs (cost, latency,
refusals) rather than where they get stuck.

**Moved to L4:** advanced sklearn (char n-grams, hyperparameter tuning, ONNX export,
`eli5`/LIME interpretability); the **username** evaluation (real product skill, but depends on the
Kaggle download and is secondary to core technical skills); local LLMs (Ollama — deliberately L4
because student hardware varies); fine-tuning; web serving; multilingual; multi-class.

---

## 4. Prototype on the private branch *first* (`kt_exploration`)

Two open questions should be settled **empirically on the private branch** before we commit to a
student-facing default. Reasoning: we shouldn't redesign the student path around an unverified
assumption — try it ourselves, measure, then decide.

### 4a. LLM provider: prototype **Gemini Flash** and compare to OpenRouter
- **Why:** OpenRouter free is still 50 req/day (unchanged, too tight). Gemini (Google AI Studio)
  advertises ~1,500 req/day free with no credit card — potentially a big friction reduction, but it
  doesn't *fully* solve rate limits and may behave differently on this task.
- **What to verify on the branch:** How hard is the SDK setup in code? Does it support **structured
  output** (huge for reliable parsing)? **Will it actually classify curse words, or refuse/filter the
  input?** (Content-safety refusals would be a dealbreaker for a profanity classifier.)
- **Findings (2026-06-11 — prototyped in `gemini_filter.py` + `gemini_smoke_test.py`):**
  - **Setup:** easy — `google-genai` SDK, ~30 lines mirroring the OpenRouter class.
  - **Structured output:** ✅ works via Pydantic `response_schema` → `response.parsed` (10/10 clean parses).
  - **Refusals:** ❌ none — with the 4 text-harm categories at `BLOCK_NONE`, it classified slurs,
    threats, `kys`, and `fucking <x>` without blocking. *Not* a dealbreaker. (Note: Gemini 2.5+
    defaults to no probability blocking anyway, but we set `BLOCK_NONE` explicitly to be safe.)
  - **Quality (toy set):** 10/10 including the Scunthorpe traps (`assessment`/`scunthorpe`/`class`)
    that substring regex & sklearn get wrong — a nice teaching contrast for L3.
  - **Latency:** ~0.54s/call on flash-lite.
  - **Rate limits (the real catch — model-dependent):** `gemini-2.5-flash-lite` = **30 RPM /
    1,500 req-day** (use this as default); `gemini-2.5-flash` = only **5 RPM** free (we hit a 429);
    `gemini-2.0-flash` is **deprecated** (June 1, 2026 — don't use).
- **Decision:** make **Gemini (`gemini-2.5-flash-lite`) the default L3 path** — 1,500/day vs
  OpenRouter's 50/day, no credit card, no refusals, clean structured output. Keep **OpenRouter as the
  optional "compare many models from one API" stretch.** The 30 RPM cap means batch evaluation still
  needs throttling + caching, so the **sample-efficiency lesson stays** (Keith's instinct that Gemini
  "helps a lot but doesn't fully solve" rate limits is confirmed).

### 4b. GameTox dataset replacement (currently blocking for new students)
- **Why this is urgent:** `github.com/shucoll/GameTox` now ships **README-only** — the CSV is gated
  behind an active [Codabench shared task](https://www.codabench.org/competitions/12083/). Keith's
  existing local `gametox.csv` still matches the 6-label schema in `data_loader.py`, but
  `data/download.sh`'s `git clone` now yields no data → the pipeline breaks for anyone starting fresh.
- **Options to try on the branch:** (1) redistribute Keith's copy for private cohort use (check
  terms); (2) have students join the shared task; (3) switch the core to a freely-downloadable
  alternative (HuggingFace toxicity set or public gaming-chat data). Each has reproducibility vs.
  fidelity vs. effort tradeoffs — prototype before choosing.
- **Progress (2026-06-11):** the data is now hosted on **Google Drive** (the shared task redirects
  there). Keith downloaded the `train/` export; `train/train.csv` has `index,message,label` (same
  6-label schema, 42,959 rows). **`data_loader.py` updated** to default to `data/train/train.csv` —
  the `index` column is ignored and float labels convert cleanly; `stats`/`regex`/`sklearn` all run.
- **Still open:** `data/download.sh` still does the now-broken `git clone`. How students *acquire*
  the data needs a decision — automate the Google Drive folder pull (e.g., `gdown`, only if the
  folder is publicly shared), have the instructor distribute the zip directly, or self-host. Tied to
  the redistribution-terms question above.

---

## 5. Cross-cutting additions to `README.md`

The student-facing brief currently has **no setup section** and no guidance on *how* to learn. Add:

1. **Getting Started** — install `uv`; a 2–3 sentence framing of **what a package manager is and why
   we use one**; `uv sync`; dataset setup; and **what a `.env` file is** and why secrets live there.
   *Reasoning:* `uv`, package managers, and `.env` were observed points of confusion. Understanding a
   package manager *is* a worthwhile skill (keep that lookup productive), but silent plumbing failures
   are pure unproductive friction — a short framing converts the latter into the former.
2. **"How to research what you don't know"** — a short meta-guide on looking up terms/concepts and
   reading docs/papers. *Reasoning:* makes the lookup itself the lesson rather than a silent gate for
   the student who doesn't know where to start. Directly targets last time's terminology struggle.
3. **"Work like a researcher"** — a hypothesis → run → measure → **log results in a table** loop,
   modeled on the existing `notes.md` results tables. *Reasoning:* directly counters "spray-and-pray"
   — presenting work should mean showing a systematic trail, not a pile of one-off attempts.
4. **Per-level self-assessment rubric** — each level gets "Done enough looks like…" and a "Stretch"
   callout. *Reasoning:* with high time-variance among students, an explicit "done enough" lets the
   time-strapped reach a real milestone and the fast ones know where stretch begins. Doubles as the
   expected-progress signpost Emily wanted.
5. **Glossary pass** — improve weak definitions; lightly trim terms whose opacity is a pointless gate;
   keep genuinely productive lookups. **Specifically disambiguate classification "positive/negative"
   (the positive class / a positive prediction) from positive/negative *sentiment*.** *Reasoning:*
   that jargon collision is a textbook example of unproductive struggle — students hear "positive" and
   think "happy/good," then misread every precision/recall explanation. Clarifying it is nearly free
   and removes a real stumbling block.

---

## 6. Per-level redesign

### L1 — Regex (core)
Keep the strong existing progression (single-word "damn" → small list → expand → evaluate → try to
bypass it → compare `alt-profanity-check`). It needs no data-science background, so it's a good
on-ramp. Introduce the **results-log/table convention here** — it's the first place metrics are
computed.

- **Target = accuracy.** *Reasoning:* accuracy is the most intuitive first metric; introduce it
  before F1 so each level adds one new idea, not several.
- **How to set the stopping point:** *we* act as smart agents and see how far we get with **only 5
  rules** — that achieved accuracy becomes the bar. *Reasoning:* a "5-rule" constraint is empirically
  grounded and achievable, and (more importantly) it forces students to confront the
  precision/recall tradeoff and think about *which* rules matter, rather than brute-forcing a giant
  wordlist. (Implementer: actually run this to fix the number — see §8.)
- Add the self-assessment rubric and experiment-loop prompts.

### L2 — sklearn (new core)
Keep the core minimal and local: `TfidfVectorizer(ngram_range=(1,1))` + `LogisticRegression`,
stratified train/test split, precision / recall / **F1**. Lean on existing sklearn tutorials.

- **Target = accuracy *and* F1.** *Reasoning:* accuracy carries over from L1 (continuity), and F1 is
  introduced *gradually* here — motivated by class imbalance (most messages are clean), which makes
  accuracy alone misleading. One new concept, well-motivated.
- **How to set the stopping point: beat the default copy/pasted sklearn pipeline.** *Reasoning:* a
  "beat the baseline you can copy/paste" goal is self-calibrating — it rewards genuine exploration
  without us hardcoding a number that may not match the (currently in-flux) dataset, and it teaches
  the real ML habit of measuring against a baseline. (Implementer: run the default pipeline to record
  the baseline numbers — see §8.)
- Move char n-grams, hyperparameter tuning, ONNX, and interpretability to L4.
- Add rubric + experiment-loop prompts.

### L3 — LLM (demo / launchpad)
Reframe as: get one classification working, then a **small, smart evaluation** — not a brute-force
run. **Fix the contradictory "thorough 50k evaluation" framing** and teach **sample-efficient
evaluation** (small/hard sets + **response caching**) as the *lesson*, not a workaround. *Reasoning:*
the rate limit isn't just an obstacle — "you can't and shouldn't brute-force an LLM over 50k rows" is
itself a real production lesson about cost/latency. Default provider pending §4a. Keep the
**1M-msg/day cost calculation** — it grounds the "when is an LLM impractical?" learning objective.

### L4 — Optional stretch
Advanced sklearn; **username evaluation** (real product skill; Kaggle-dependent); local LLMs (Ollama);
fine-tuning (ModernBERT/DistilBERT); web serving (keep `notes.md`'s honest Transformers.js failure
write-up as a lesson in itself); multilingual; multi-class. *Reasoning:* all valuable, none should
block core technical skills, and several (local LLMs, web serving) carry hardware/setup variance we
don't want on the critical path.

---

## 7. Reference-branch code work (`kt_exploration`)

- **Complete the L3 (LLM) reference** in `llm_filter.py`: add an `evaluate` command that runs on a
  small GameTox sample and reuses `print_evaluation_report` from `evaluation.py`; add **response
  caching** (`diskcache`); optionally add **batched** classification (~100 msgs/request) to
  demonstrate sample-efficiency. *Reasoning:* it's the one materially incomplete reference, and the
  new L3 framing (small smart evals + caching) needs a working example to teach from.
- **Verify the whole branch runs end-to-end** against a working `gametox.csv` once §4b lands.

---

## 8. Setting the concrete targets (an empirical task, not a guess)

The targets in §6 are deliberately defined *operationally* rather than as magic numbers, because the
dataset is in flux (§4b) and hardcoded thresholds would rot. Before finalizing student-facing copy:

1. **L1:** hand-build a 5-rule regex filter, run `regex evaluate`, record the **accuracy** → that's
   the L1 bar (or "get within X of it").
2. **L2:** run the default copy/paste `TfidfVectorizer + LogisticRegression` pipeline, record
   **accuracy + F1** → that baseline is the L2 "beat this" bar.

Record both in `notes.md` so the numbers are reproducible and the rationale is preserved.

**Initial reference points (2026-06-11, new GameTox train split — 42,959 rows, 19% toxic binary):**
- *Regex, built-in 9-word list:* accuracy **0.821**, profane F1 0.139 (precision 0.824, recall
  0.076). High precision, near-zero recall — a vivid "wordlists barely catch anything" baseline.
  (The L1 bar should still come from *5 hand-tuned rules acting as smart agents*; this 9-word number
  is just the current built-in reference.)
- *sklearn default pipeline (TF-IDF unigrams + LogisticRegressionCV):* accuracy **0.905**, profane
  F1 **0.711** (precision 0.843, recall 0.614), macro-F1 0.827. **This is the L2 "beat this" bar.**

---

## 9. Verification

- **Dataset gate (blocking, see §4b):** confirm a working `gametox.csv` before claiming any level runs.
- **Run-through:** with data present, exercise the reference CLI — `uv run main.py stats`,
  `regex evaluate`, `sklearn train`/`evaluate`, and the new `llm evaluate` — and sanity-check metrics.
- **Freshness:** validate any LLM model IDs referenced still resolve; confirm the Kaggle usernames
  `curl` path (likely needs auth) and document the real steps; confirm `requires-python` / dependency
  pins install cleanly via `uv sync`.
- **Pedagogy check (manual):** read each level as a *time-strapped* student. Is the productive
  struggle intact and the unproductive friction (toolchain / data / opaque terms / scope) removed?
  Does the rubric give a clear "done enough"?

---

## 10. Open items

- Outcome of the **Gemini vs. OpenRouter** prototype (§4a) → sets the L3 default.
- Outcome of the **GameTox** decision (§4b) → unblocks the data path.
- The **recorded** L1 accuracy bar and L2 accuracy+F1 baseline (§8).
- Any further terminology collisions to disambiguate beyond positive/negative-vs-sentiment.
