# Tek Innov8rs: Profanity Filter

## Context
This project is based on real-world work done at Singularity 6 for the game [Palia](https://palia.com/), a multiplayer online game available on PC/Steam, Switch, Xbox, and PlayStation.

### Project Goals
In a production environment, a profanity filter needs to balance multiple objectives:
- **Player safety**: Prevent and reduce harm from hate speech, harassment, and other toxic behavior through automated detection
- **Industry standards**: Follow gaming conventions where profanity is automatically censored
- **Legal compliance**: There are many proposed regulations about online chat safety, particularly for children
- **Platform requirements**: Satisfy certification requirements from Nintendo, Microsoft, and Sony. Microsoft's requirement is publicly available at [XR-018](https://learn.microsoft.com/en-us/gaming/gdk/docs/store/policies/xr/xr018)

### Real-World Constraints
The production system had to handle:
- **Global scale**: Support for 10+ languages including English, German, Spanish, Dutch, Portuguese, French, Russian, Chinese, Japanese, and Hungarian
- **Cross-language profanity**: English curse words frequently appeared in other language contexts
- **Multilingual servers**: Chat rooms mixed players from different language backgrounds
- **High volume**: Approximately 1 million chat messages per day

### User Experience Considerations
- **Username filtering**: Blocked usernames need clear explanations (e.g., "cass" blocked due to substring "ass")
- **Visual feedback**: Star out flagged words in chat (e.g., `***essment`) so users can identify false positives
- **Adversarial behavior**: Users actively attempt to bypass filters using leetspeak, misspellings, and Unicode characters

## What's changed (June 2026)

If you took this course in the previous run, here's what's different — and why:

- **The levels were reordered.** Traditional ML (scikit-learn) is now **Level 2** and LLMs moved to
  **Level 3**. Levels 1–2 are the core; Level 3 is a guided exploration; Level 4 is optional
  stretch. *Why:* last run, students who reached the LLM level often got stuck there — partly on
  external friction (API keys and rate limits), and partly because **good subjective evaluation of
  an LLM is hard without first having a foundation in objective evaluation**. The sklearn level
  turned out to be a smoother experience for those same reasons, and because there are so many good
  online guides for it. So it now comes first, where you build the evaluation intuition you'll lean
  on when you get to the LLM.
- **Level 2 (sklearn) is leaner; the deeper ML work moved to Level 4.** Advanced exploration —
  character n-grams, hyperparameter search, model interpretability, ONNX export — is now optional
  Level 4 stretch rather than part of the core. *Why:* keeping the core tight means more students
  actually make it all the way through Levels 1–3, instead of stalling partway. The depth is still
  there for anyone who wants it; it just no longer gates the core path.
- **The LLM level now uses Google Gemini** instead of OpenRouter. *Why:* OpenRouter's free tier
  (50 requests/day) was a real, concrete blocker last run — students simply ran out of requests
  before they could finish their work. Gemini's free tier (~1,500 requests/day, no credit card)
  removes that wall.
- **The dataset is downloaded differently.** GameTox is now distributed via its
  [Codabench shared task](https://www.codabench.org/competitions/12083/) / Google Drive (the old
  GitHub repo is README-only now). See [Getting Started](#getting-started).
- **New onboarding and guidance.** Added Prerequisites, a Getting Started section (package manager,
  dataset, `.env` files), and two how-to-learn guides — "Researching what you don't know" and "Work
  like a researcher" — plus a "Definition of Done" / "Check yourself" rubric on each level. *Why:*
  to cut down the time spent struggling on incidental, non-essential plumbing (toolchain, setup,
  finding your footing) so that more of your time and attention goes to the lessons that actually
  matter.
- **How this revision was made.** I used Claude Opus to help edit these docs and expand the
  reference code on the private branch — a way to balance making meaningful improvements against my
  limited schedule. I reviewed and directed the changes; flagging the provenance for transparency.

## Prerequisites

This project assumes you're comfortable with:
- **Python** — reading and writing small scripts.
- **Package managers** — you don't need to be an expert, but you should understand what a package manager *is* and why a project uses one. If you've used `pip`, `poetry`, `conda`, or `npm`, you're set. If package managers are new to you, spend a few minutes reading about one (I recommend [`uv`](https://docs.astral.sh/uv/)) before starting.

## Getting Started

You'll build this project yourself. To get going:

1. **Set up a Python project** with a package manager. Add libraries as you need them — `pandas` and `scikit-learn` for the data/ML levels, an LLM SDK for the LLM level.
2. **Get the GameTox dataset** (labeled gaming chat) from its [shared task](https://www.codabench.org/competitions/12083/), currently hosted in a public [Google Drive folder](https://drive.google.com/drive/folders/1HkfwexOpX1S9gRrMeCFMfZJjsBs0hQRu). **Download `train.csv` directly from that folder in your browser** (columns: `index, message, label`) and put it somewhere sensible like `data/GameTox/`. Name dataset folders clearly and note where they came from — you may add more later. It's also good to add a README.md into the folder with any notes like where they came from or what's in the data.

(API keys come later, in Level 3, when you first need one.)

## Project Overview
You'll build a simplified version of a production profanity filter, implementing progressively more sophisticated approaches. All code should be written in Python, primarily as CLI scripts.

The levels build up in sophistication: **rules → traditional ML → LLMs**. Levels 1 and 2 are the core of the project; Level 3 is a guided exploration of LLMs; Level 4 is optional stretch work. A big part of the learning is comparing these approaches — accuracy, speed, cost, and effort — so you understand when each is the right tool.

### Learning Objectives

- **Solution spectrum**: Understand tradeoffs between rule-based systems, traditional ML, and LLMs. When does each approach excel?
- **Text classification**: Gain hands-on experience with applied machine learning for NLP tasks
- **LLM integration**: Learn to work with LLM APIs effectively
- **Text processing**: Handle real-world challenges with Unicode, multilingual data, and noisy text
- **Realistic, messy problems**: Work with real-world data that's far messier than a typical class assignment — incomplete labels, ambiguous cases, and rarely one clean "right answer".
- **Evaluation, not just testing**: Learn to *measure* quality with metrics like precision and recall. This is a different mindset from unit testing, where code is simply right or wrong — here, every approach is partly wrong, and the question is *how* wrong and in which ways.

### How to Work on This Project

**Researching what you don't know**

This project uses real terms and tools without defining all of them. Looking things up — starting with official docs — is part of the work, not a sign you're behind.

**Work like a researcher**

Most levels ask you to improve and compare approaches. The naive loop is to randomly change settings and keep whatever scores highest — don't do that. The real skill is forming a *hypothesis* about what will help, and a good hypothesis comes from **looking at your errors**:

- **Inspect what your filter gets wrong.** Pull up the false positives and false negatives and actually read them. What do they have in common? Concatenated words? A misspelling? A word missing from your list? Sarcasm or context the model can't see?
- **Turn the pattern into a hypothesis.** For example: "Lots of my misses are multi-word insults → maybe matching adjacent *pairs* of words will help." But you can only hypothesize about solutions you know exist — and early on, you mostly won't. That's why reading the documentation matters: skimming the scikit-learn `TfidfVectorizer` docs reveals the easy knobs you could turn (word vs. character n-grams, document-frequency cutoffs, stop words); reading up on LLM prompt engineering reveals techniques like chain-of-thought, few-shot / in-context learning, and structured outputs. Knowing the solution space is what turns a vague itch into a testable hypothesis.
- **Change one thing, then measure** on the same data with the same metric — so you can actually tell whether it helped.
- **Log every run** — what you changed, *why*, and the resulting numbers — in a small table. Keep the failures, and describe them precisely: "I tried X because Y; it fixed some of the cases I was targeting but introduced new false positives on Z" is far more useful than "it didn't help." The *direction* of the tradeoff is the real result.

When you present your work, walk through that trail: here's what I saw in the errors, here's what I tried and why, here's what happened. That's far more convincing — and more useful — than demoing the one thing that worked.

### Level 1: Rule-Based Filter

Build a profanity filter the simplest way possible: matching messages against a list of bad words. The point isn't to build a *great* filter — it's to feel firsthand where simple rules break down, and to get comfortable with the evaluation metrics you'll use for the rest of the project.

**Tasks:**
1. Get the data: you'll use the **GameTox** dataset (see [Getting Started](#getting-started)). It has a `message` column and a `label` column (0 = non-toxic, 1–5 = different kinds of toxic). For now, treat it as binary: toxic (`label > 0`) vs. clean.

2. Start simple — single-word detection:
   - Write a script that counts what percentage of GameTox messages contain "damn"
   - Manually review some of those messages — are they all actually toxic?
   - Count how many messages your script flags (all messages with "damn")
   - Count how many of those are actually labeled as toxic in GameTox (correct flags)
   - Count how many are labeled as not toxic (incorrect flags)

3. Build a regex-based profanity detector:
   - Start with a small list of profane words (5–10 words)
   - Create a regular expression that matches any of those words
   - Test it and observe the same counts as above: total flagged, correct flags, incorrect flags
   - Expand your word list (download a profanity list or grow your hand-written one)
   - Binary classification: profane vs. clean (ignore GameTox's multiple categories for now)

4. Formal evaluation:
   - Calculate accuracy, precision, and recall on GameTox data
   - Try to bypass your own filter with creative misspellings — what gets through?

5. Baseline comparison:
   - Compare against [alt-profanity-check](https://github.com/dimitrismistriotis/alt-profanity-check)

**Definition of Done**

This level is less about a great filter than about what you walk away understanding. You're done when you can check off:

*Evaluation results*
- You report **accuracy, precision, and recall** on GameTox, with a concrete example of both a false positive (e.g., "assessment") and a false negative that you found by reading the data.
- With ~5 well-chosen rules you reach **accuracy in the mid-80s% (around 85%)**. Notice the bar, though: ~81% of GameTox messages aren't toxic, so a filter that flags *nothing* already scores ~81% — only a few points below your rules, even though they're catching real profanity.

*Engineering*
- You have a regex-based filter you can run over GameTox and rerun as you expand your word list.
- You've compared it against [alt-profanity-check](https://github.com/dimitrismistriotis/alt-profanity-check) on the same data.

*What you should take away*
- You've felt the classic **precision/recall tradeoff** firsthand: blocking more offensive content inevitably catches some innocent text.
- You understand why **accuracy alone is misleading** on imbalanced data — that tiny accuracy gap above is the whole reason Level 2 switches to F1.

*Stretch (optional)*
- Grow and curate a larger word list, and hunt for creative misspellings/obfuscations that slip past it — motivation for the later levels.

**Key Learnings:**
- Regular expressions
- Working with real-world datasets (incomplete documentation, inherent biases)
- Evaluation metrics (accuracy, precision, recall) vs. traditional testing approaches

**Terminology**
- **Accuracy**: The percentage of predictions that are correct (both positive and negative). Can be misleading with imbalanced datasets.
- **"Positive" vs. "negative" (in classification)**: These have *nothing* to do with sentiment or good/bad. "Positive" means *the thing you're detecting* — here, a toxic/profane message — and "negative" means everything else (clean). So a "false positive" is a clean message wrongly flagged, and "recall" is the fraction of truly-toxic messages you caught.
- **False positive**: An item incorrectly classified as positive (e.g., flagging "assessment" as profane).
- **False negative**: An item incorrectly classified as negative (e.g., missing an actual profane message).
- **Precision**: Of all items flagged as positive, what percentage are actually positive? High precision means few false alarms.
- **Recall**: Of all actual positive items, what percentage did we catch? High recall means we don't miss much.
- **Regular expression**: A pattern-matching language for finding text sequences (e.g., `\b(bad|worse|worst)\b` matches those exact words).
- **Binary classification**: A task with exactly two possible outcomes (e.g., profane vs. clean).

### Level 2: Traditional ML Classifier

Instead of hand-writing rules, let a model *learn* what toxic language looks like from the labeled data. This is the workhorse of practical text classification: it runs locally on any hardware, there are excellent tutorials, and it's the best place in this project to get a real machine-learning result.

**Tasks:**
1. Train a scikit-learn text classifier:
   - Split the GameTox `train.csv` into your own training and test sets. (Yes — you're splitting the file that's *named* `train`. That's normal: you need a held-out slice the model never sees so you can measure how it does on unseen messages.) Use a **stratified** split so both sets have a similar toxic ratio.
   - Start with features from `TfidfVectorizer(ngram_range=(1, 1))`
   - Start with a `LogisticRegression` classifier
2. Evaluate properly:
   - Measure accuracy, precision, recall, and **F1** on the held-out test set
   - Compare against your Level 1 rule-based filter on the same data
3. Establish a baseline, then improve on it:
   - First, run the pipeline with **default settings** (`TfidfVectorizer(ngram_range=(1, 1))` + a plain `LogisticRegression`) and evaluate it. That result is *your* baseline — the number to beat.
   - Then try to improve on it — experiment with features and hyperparameters (see tips below), changing **one thing at a time**, and **log each run's metrics** (see [Work like a researcher](#how-to-work-on-this-project)).

**What to expect:** the untuned default already reaches roughly **0.90 accuracy** — but on imbalanced data (~19% toxic) accuracy flatters you, so the number that actually matters is **F1 on the toxic class**, which starts much lower (around **0.67** out of the box). Don't chase a fixed target you copied from someone else; measure your own default, then beat it — and watch F1, not accuracy, as you tune.

> **Why F1 now?** Because the data is imbalanced (~19% toxic), accuracy can look high while the model misses most toxic messages (you saw this in Level 1). **F1** — the harmonic mean of precision and recall on the toxic class — is the honest single number to optimize here.

**Helpful Resources:**
- [scikit-learn text classification tutorial](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- [GeeksforGeeks NLP classification guide](https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/)

**Tips for beating the baseline:**
- `TfidfVectorizer`: try `ngram_range=(1, 2)`, different `min_df` values, or `analyzer='char'` vs `'word'`
- `LogisticRegression`: try different `C` values
- If it's running slowly, increase `min_df` to shrink the model — this also helps prevent overfitting (though it can throw away some useful words)

**Key Learnings:**
- Train/test methodology and avoiding overfitting
- Precision, recall, F1-score, and when to optimize for each
- Traditional ML as a practical middle ground between rules and LLMs

**Terminology**
- **Training vs testing data**: Training data is used to build the model; testing data evaluates how well it generalizes to unseen examples.
- **Held-out data (same as testing data)**: Data deliberately set aside and not used during training, reserved exclusively for evaluation.
- **F1-score, F-score, F-measure**: The harmonic mean of precision and recall, balancing both metrics. Commonly used instead of accuracy when the positive class is rare (most messages aren't toxic).
- **Stratified split**: A train/test split that preserves the class proportions (e.g., keeps ~19% toxic in both sets), so evaluation isn't skewed by an unlucky split.
- **Document frequency**: How many documents contain a particular word. Rare words (low DF) are often more informative than common ones.
- **TF-IDF**: Term Frequency–Inverse Document Frequency; weights words by how often they appear in a document relative to how rare they are overall. Reduces the impact of common words like "the".
- **Stop words**: Extremely common words ("the", "a", "is") that are often removed because they add little meaning. May or may not help depending on your task.
- **Ngram**: A sequence of N consecutive words (or characters). Bigrams (2-grams) like "very bad" can capture meaning that single words miss.
- **Logistic regression**: A simple but effective classification algorithm that learns weights for features and outputs a probability. Despite its name, it's used for classification, not regression.
- **Hyperparameter**: A setting that affects how a model is trained. For `LogisticRegression`, `C` is a key example; for `TfidfVectorizer`, `min_df` and `ngram_range` matter a lot.

**Check yourself**
- *Done enough:* a trained classifier evaluated on a held-out test set, with accuracy/precision/recall/F1 reported and compared to your Level 1 filter — and at least one experiment that tried to beat the default, logged in a results table.
- *Stretch:* meaningfully beat the default F1 through systematic tuning (see Level 4 for going further with char n-grams, grid search, and interpretability).

### Level 3: LLM-Based Filter

Now try the heavy hitter: a large language model. The interesting question isn't "*can* an LLM do this?" (it can) — it's *when an LLM is actually the right tool*, given cost, latency, and rate limits. Treat this level as a guided exploration rather than a full build.

**Use a generous free tier.** I recommend Google's Gemini via [Google AI Studio](https://aistudio.google.com/apikey): the free tier allows ~1,500 requests/day with no credit card, and the **`gemini-2.5-flash-lite`** model is fast and more than capable for binary classification. (Avoid `gemini-2.5-flash` on the free tier — it's throttled to ~5 requests/minute.)

**API keys and `.env`.** Get a free key from [Google AI Studio](https://aistudio.google.com/apikey). Secrets like API keys don't belong in your code: a key hardcoded in a script is easy to commit by accident, and once it's in git history it stays there even after you "delete" it — and if you ever push to GitHub, bots actively scan public repos for leaked keys and will grab yours to burn through your quota or rack up charges on your account. Instead, put it in a `.env` file at your project root, add `.env` to your `.gitignore`, and load it at startup with [`python-dotenv`](https://pypi.org/project/python-dotenv/). A `.env` is just a plain text file of `KEY=value` lines (e.g., `GEMINI_API_KEY=...`). Never commit it.

**Work with a small, hand-picked set of messages.** On a free tier you can't run an LLM over all ~43,000 messages — it's slow and the daily quota is tight — so here you'll work with a *small* set (think ~30–50 messages), chosen on purpose:
- **Pick hard examples** — messages where your Level 1 and Level 2 filters disagree, or that they get wrong. These are far more revealing than easy or random ones. Finding them usually means **writing a bit of code**: run both filters over the data and pull out the messages where they disagree.
- Use the *same* messages to look at all three approaches (rules, ML, LLM) side by side.

> **How this differs from industry.** The tiny set is a *teaching* constraint from the free tier, not a best practice — don't take "~50 messages" as the professional norm. With a paid API you'd still *start* small and subjective (eyeball ~30–50 hard cases while iterating on the prompt), but once the prompt looks good you'd run a formal eval on ~1,000 messages, and only run the full ~43,000 for a final number you trust. The instinct is the same — start small, scale as your confidence grows — but in industry cost no longer forces you to stop at 50.

**Evaluate by reading, not by metrics.** This set is small and hand-picked, so precision/recall/F1 wouldn't be trustworthy here (a biased handful of messages can't give reliable numbers). Instead, evaluate *subjectively*: read each message alongside each filter's verdict and form your own judgment about which approach got it right, and why.

**Tasks:**
1. Set up access: install the official [`google-genai`](https://ai.google.dev/gemini-api/docs/quickstart) SDK and get a free Gemini API key (stored in `.env`, above). Call Gemini with a clear prompt for binary (profane/clean) classification, using [**structured / JSON output**](https://ai.google.dev/gemini-api/docs/structured-output) so responses are reliable to parse.
2. Build your hard set: write code to find messages where your Level 1 and Level 2 filters *disagree*, and collect ~30–50 of them.
3. Compare by hand: run all three filters (rules, ML, LLM) on that set and read through the results. Where does the LLM win? Where does it lose? What does it catch that your other filters miss (sarcasm, context, obfuscation), and where does it overreach?
4. Production feasibility: estimate the cost and time to classify **1,000,000 messages/day** with a paid model. Is an LLM practical at that scale? Compare against your Level 2 classifier.

**Key Learnings:**
- Calling an LLM API and designing a prompt for classification
- Working within rate limits — getting real signal from a small, hand-picked set of hard cases
- Judging quality *subjectively* when formal metrics don't apply
- Cost / latency / rate-limit tradeoffs, and when an LLM is (and isn't) practical

**Terminology**
- **Prompt engineering**: Designing and refining the text instructions you give an LLM. Small wording changes can noticeably change accuracy.
- **Structured output**: Asking the LLM to respond in a fixed format (like JSON), making its answers reliable to parse instead of free-form text.
- **Rate limit**: A cap on how many requests you can make per minute or per day. Free tiers are limited, so designing around this (small sets of messages) is part of the job.
- **Latency**: The delay between sending a request and getting a response. Critical for real-time chat filtering, and usually much higher for LLMs than for a local model.

**Check yourself**
- *Done enough:* a working LLM classifier run on a small, hand-picked set of hard messages, compared *by eye* against your Level 1–2 filters on that same set, plus a back-of-the-envelope cost estimate for 1M messages/day.

### Level 4: Advanced Directions (optional)

Optional stretch work — pick whatever interests you. None of this is required; it's where to go if you've finished the core levels and want more.

**Go deeper on the ML classifier (Level 2):**
- Package your model in a scikit-learn `Pipeline` for cleaner deployment. A real test of "packaging": can you send the *trained* model to a friend and have them run it on new text **without retraining it themselves**? That sounds trivial, but it's surprisingly hard — you have to ship the fitted vectorizer and model together and load them on the other end.
- Systematically tune hyperparameters with [grid or random search](https://scikit-learn.org/stable/modules/grid_search.html)
- Understand *why* the model predicts what it does with [`eli5.show_weights`](https://eli5.readthedocs.io/en/latest/autodocs/eli5.html#eli5.show_weights) ([example](https://gist.github.com/jantrienes/13c53b841cdb98f3aaaf5f7147df7a23)) or [LIME](https://github.com/marcotcr/lime)
- Export your model to ONNX for fast, dependency-light inference

**Username filtering (a real product problem):**
- Run your filters over the [Reddit Usernames dataset](https://www.kaggle.com/datasets/colinmorris/reddit-usernames) (unlabeled, real usernames). Review the flagged ones to estimate precision.
- Usernames have no surrounding context and often concatenate words ("ihatethisgame"), so word-boundary tricks fail. Experiment with handling case transitions and substrings — and notice the false positives (e.g., a "cass" or "assessment" problem). Expected offensive rate in the wild: ~0.1–5%.

**Make the LLM practical (Level 3):**
- Add response **caching** (e.g., [diskcache](https://grantjenks.com/docs/diskcache/tutorial.html)) so duplicate/repeat messages don't re-hit the API — cutting average latency and cost
- **Batch** many messages into one request to get more out of a daily quota
- Run a small model **locally** with [Ollama](https://ollama.com/) — no API limits. Start with a small model
- Extend to **multi-class** classification (clean / profanity / insult / hate speech)

**ML/AI Approaches:**
- Fine-tune a transformer model like [ModernBERT](https://huggingface.co/blog/modernbert) on your dataset. **Heads up:** fine-tuning is memory- and compute-hungry, and on modest or older hardware you'll likely hit out-of-memory errors or painfully slow training. Working through that is part of the exercise — but if you get truly stuck, the levers that help (a smaller/"tiny" model, smaller batch size, shorter input length, iterating on a data subset) are written up in [`docs/2025-12-questions-and-answers.md`](docs/2025-12-questions-and-answers.md).
- Benchmark against pre-trained models like [toxic-bert](https://huggingface.co/unitary/toxic-bert)
- Explore censoring (****ing) via token-level approaches (BERT) vs. generative approaches (T5)

**Expanded Support:**
- Extend to non-English languages using multilingual BERT or LLMs
- **Important**: Validate data quality carefully for languages you don't speak
- Evaluate on additional datasets (see `docs/2025-11-perplexity-note.md` for options)

**Deployment & Engineering:**
- Package your solution as an installable Python library
- Build a web API for real-time filtering

**Terminology**
- **Multi-class**: Classification with more than two categories (e.g., clean / profanity / insult / hate speech), as opposed to binary.
- **Fine-tuning**: Taking a pre-trained model and continuing to train it on your specific dataset. Leverages existing knowledge while specializing to your task.
- **Transformer**: A neural network architecture using attention mechanisms to process sequences. The foundation for BERT, GPT, and most modern LLMs.
- **Generative**: Models that produce new text rather than just classifying existing text. Can "rewrite" profane messages into clean versions rather than just detecting them.
