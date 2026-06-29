# Q&A

## How chat works

Follow-up from last night on "how chat works."

I'm pretty sure Palia used [Vivox](https://docs.unity.com/en-us/vivox-unreal) as the chat provider in 2024, and they probably still do. For context on the codebase: Palia was written in Unreal Engine for both the game client and the game servers, with backend web services in Rust. I'm hazy on some of these details, but this should clear a few things up.

### Setup

Some setup has to happen before chat works:

1. The player's client logs in, which triggers the backend to log in to Vivox. Vivox sends a token back to the client.
2. The client signs in to the Vivox servers to receive messages — a long-lived connection under the hood. On sign-in, it registers for the channels it wants messages on (e.g. `Server15_General`, `Server15_RegionalChat`, `Guild_1234`).

### Sending a message

1. The player hits enter, which makes an API call to the Palia chat backend.
2. The backend censors profanity, then sends the censored message to the Vivox servers on the player's behalf.

### Receiving a message

1. The Vivox API in the receiver's Palia client triggers a callback.
2. (Nintendo Switch only) An additional Nintendo-specific profanity filter runs on-device, required by Nintendo.
3. The Palia Unreal code adds the message to the UI.

### Things I'm not 100% clear on

- **Vivox's own profanity filtering.** Vivox has some, but I don't remember much discussion of it, so I'm not sure why we didn't use it — except that we'd already built our own filter before switching our chat service to Vivox.
- **Token security.** I'm not entirely sure if the game client gets a read-only token from Vivox or a read-write token, though I'm pretty sure the client itself is only reading.

## Hyperparameter tuning on TF-IDF + LogisticRegression — does it actually move the needle?

Yes, it helps. See the note in [reference results](../reference/docs/2026-06-reference-results.md).

The basic sklearn model with defaults achieves 0.67 F1 on profane messages. Tuning the hyperparameters of the TfidfVectorizer and swapping out LogisticRegression for LogisticRegressionCV to tune the `C` value brings it up to 0.71.

Further improvements are possible with character ngrams.

## Why Random Forest helped, but switching LR → Multinomial Naive Bayes didn't

> I narrated to Claude for this section; it's not verbatim what I said but a better organized version.

Context: A student saw little change moving from Logistic Regression to Multinomial Naive Bayes, but got a real improvement from a Random Forest. Here's the intuition for why.

**Naive Bayes is, for our purposes, just another linear model** — like Logistic Regression, it
scores a message by adding up a contribution from each word. In my experience, linear models tend to
land in roughly the same ballpark on a task like this. Where they differ is mostly in their
*regularization* and *inductive bias*, and that gap mostly shows up when you have very little
training data. With a decent-sized dataset like GameTox, swapping one linear model for another
usually isn't where the gains are — so it's no surprise LR → NB barely moved.

Terms:
- **Regularization**: the techniques a model uses to avoid *overfitting* (fitting the training data
  so closely that it memorizes quirks and noise, and then does worse on new messages). Different
  linear models hold themselves back from overfitting in different ways, which is why they can
  disagree when data is scarce.
- **Inductive bias**: the assumptions a model makes by default about what a good answer looks like.
  Two models given the exact same data can reach different conclusions because they "lean" in
  different directions out of the box.

**Random Forest is different in two ways that matter here:**

1. **It's well-regularized by default.** A Random Forest trains many decision trees and averages
   them, deliberately varying what each tree sees:
   - **Bagging**: each tree is trained on a different random resample of the messages.
   - **Feature subsampling**: each tree is only allowed to split on a random subset of the
     words/features.

   Averaging lots of trees that were each shown a different slice smooths out the noise any single
   tree would latch onto — so you get sensible regularization for free, without tuning for it.
2. **It can use *combinations* of features.** A linear model weighs each word on its own. A tree can
   branch on "this word *and* that word," so a forest naturally picks up that certain *pairs* or
   *triples* of words together signal toxicity even when the individual words don't. (You can hand a
   linear model some of this by adding n-gram features, but the forest gets it without being asked.)

## What would I actually do in practice? (choosing an approach)

> I narrated to Claude for this section; it's not verbatim what I said but a better organized version which I edited afterwards.

The honest answer is *it depends* — on time, budget, the number of players and
messages per day, the languages the game is played in, latency requirements, how much labeled data
you have, what your backend can actually run, privacy, and who has to maintain the thing afterward.
Here's roughly how I'd work through it, from the simplest option upward.

### Start simple: an off-the-shelf library

I'd reach for an existing profanity-filter library in whatever
language our backend is written in — if one exists — and start there. It's low-latency, quick to
run, and easy to integrate and maintain.

Even for a global launch where I genuinely don't know what to expect, this is still my default
starting point: start with the off-the-shelf library, see whether it holds up across the languages the players actually speak, and observe how it works with real usage. 

### Why I'd probably *not* start with an LLM

I considered a small LLM like a gpt nano model, but decided it just doesn't make sense for many scenarios. The biggest problem is that you don't know how big your player base will be before you launch. Maybe it'll be ten people, maybe it'll be a million. The spike from ten to a million could happen overnight if a popular streamer tries it on stream, and it's too risky to have unbounded LLM costs especially for a small studio.

The second problem is that some gamers can be privacy-conscious, so they aren't going to be thrilled about you donating their data to AI vendors. It's just not worth the risks.

### With more time, budget, and data: use an LLM to *annotate*, then train a small model

The next level I'd explore: write a strong LLM prompt, then **offline** annotate something like
10,000–500,000 chat messages with it, and train a small scikit-learn model on those labels. If the profanity in game is relatively simple, this often captures about 80% of the quality of the LLM but it's faster, cheaper, and more private.

The constraint to check first: can the system that runs the filter actually run a scikit-learn model?
(More on that in the Palia story below — it's exactly where this can fall down.)

### Keeping an LLM in the loop without serving it live

I wouldn't call an LLM on every live message, but if you want LLM-level judgment at serving time,
**semantic caching** is an option. See [this](https://www.percona.com/blog/semantic-caching-for-llm-apps-reduce-costs-by-40-80-and-speed-up-by-250x/) for an example. I haven't used this approach myself, partly because it doesn't solve the unbound costs or privacy problem, but it's a common approach in industrial LLM applications.

### Fine-tuned BERT

We explored a fine-tuned BERT-style model at Palia, and they're genuinely good toxicity classifiers. The tradeoff is model size and CPU/GPU demands: it can be fine to run on a server, but prohibitive to run inside a game client.

### What we actually did at Palia — and why

The backend services were written in Rust, and the team started with a Rust profanity-filter library. It worked fairly well in English and was extensible — you could add bad words or add exceptions. 

What I did was take that Rust filter and tune it to work well not just in English but across roughly the top 10–20 languages. The hard part wasn't the code — it was finding **test data**. I split real user messages by language and set an expectation per language (say, ~0.1–1% of messages contain profanity). If a language was flagging far more than that, we probably had a false-positive problem; far less, a false-negative problem. That triage told me which languages to review and tune.

Why not the scikit-learn model, which probably would have been more accurate — even multilingually — than what we shipped? Because the backend was Rust and scikit-learn is Python. Exporting a model from Python to Rust isn't impossible, but it's a complex pipeline, and it would have tied the filter to our Python/Databricks data ecosystem with a model loaded on the Rust side. The engineers maintaining the system were Rust-only and realistically couldn't maintain that. The data-science team could have, but we weren't full-time on the filter — we'd work on it for a bit, then move on to something else.

Picking something aligned with the team's skills and maintainable turned out, in retrospect, to be a really good decision. The data-science team eventually got stripped down to almost nobody — there were no ML people after me — so having a filter a backend engineer could read, understand, and maintain mattered a lot. Bringing testing and evaluation *into* the backend library itself was part of that.

**The lesson:** the model that wins on a metric isn't automatically the right production choice.
Maintainability, the team's skills, runtime and language constraints, cost, and privacy often decide
it instead.

### If we'd had the time: what doing it *really* well would look like

Everything above was shaped by constraints. If instead we'd had lots of time and the goal was simply to do a *great* job of profanity filtering, here's where I'd take it. The two surfaces — usernames and chat — have genuinely different needs, so I'd treat them as related but distinct problems.

**Transparency is a real product requirement, especially for usernames.** When we reject a username, we need to tell the player *why*, as clearly as we can — a concrete example is rejecting "assassin" because it contains the substring "ass." We generated a lot of support tickets from exactly this: players trying to register the username they use in every other game, getting rejected, and not being told why. So a core design need is to surface the reason for a rejection — to the player when possible, and to our support team at the very least.

**Chat messages: a character-level seq2seq "censoring" model.** For chat, the ideal behavior is just to star out the offensive words (`****`). The best version I can picture is a **character-based sequence-to-sequence model**: it takes the message as a character sequence (optionally with a little context like the recent messages, since profanity and especially hate speech can be contextual) and outputs the same message with the genuinely offensive parts starred out. Training that needs a lot of data — we could build it from high-quality off-the-shelf libraries, or generate it with a large, advanced LLM.

I'd search for a pretrained character-based T5 variant for this, because T5 have a history of being well-adapted to many tasks.

**Usernames: classify, and point at the offending span.** Usernames behave differently (no spaces), so if I could find a T5-like model I'd set the task type to this new one and train on usernames. Ideally they would be real usernames with character-level censoring done by a strong LLM. Then when someone tries to sign up with a bad username we could reject it by comparing the original username and the model output for differences, and highlight the bad parts.

## Grid search and systematic hyperparameter tuning

> Narratived to Claude then simplified.

Context: with so many knobs to try (`C`, `ngram_range`, `min_df`, analyzer, ...), tuning can feel like flailing. The fix is to make it **systematic** instead of guess-and-check. (*Hyperparameter* is defined in the Level 2 terminology in the README — the settings like `C`, `min_df`, and `ngram_range` that control how the model trains.)

**Grid search** is the simplest way to be systematic: you list the values you want to try for each hyperparameter and evaluate every combination — conceptually just a set of nested for-loops, except scikit-learn handles the bookkeeping and cross-validation for you ([docs](https://scikit-learn.org/stable/modules/grid_search.html)).

**The goal is understanding, not just picking the winner.** A lot of tuning by hand is really about figuring out *which* hyperparameters actually move your metric, so you can focus effort there and stop fiddling with the ones that don't. Two habits help:
- **Read the trend, not just the top score.** If you sweep `C` over `1.0 … 100.0` and the best value comes back as `100.0` — the very edge of your range — that's a signal to try *beyond* it (`200`, `500`), especially if the metric is still climbing meaningfully. A winning value sitting on the boundary is telling you the range was too narrow. 
- **Look at the whole set of results, not one number.** Pulling every combination's score into a table (e.g. pandas) and eyeballing metric-vs-hyperparameter makes it obvious which knobs matter, which are flat, and where the gains plateau.

The mindset shift is the lesson: from **guess-and-check** ("change something, keep it if the number went up") to **systematic** ("sweep a range, see how the metric responds, follow the trend").

> **Footnote — beyond grid search.** Grid search gets expensive fast as you add hyperparameters,
> since every combination multiplies the others. For those cases there are smarter strategies —
> **random search** (sample random combinations; surprisingly effective when only a few
> hyperparameters actually matter) and other scikit-learn search methods — worth reaching for once a
> full grid is too big to run.