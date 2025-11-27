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

## Project Overview
You'll build a simplified version of a production profanity filter, implementing progressively more sophisticated approaches. All code should be written in Python, primarily as CLI scripts (Jupyter notebooks are acceptable for exploration and analysis).

### Learning Objectives

- **Solution spectrum**: Understand tradeoffs between rule-based systems, traditional ML, and LLMs. When does each approach excel?
- **Text classification**: Gain hands-on experience with applied machine learning for NLP tasks
- **LLM integration**: Learn to work with LLM APIs effectively
- **Text processing**: Handle real-world challenges with Unicode, multilingual data, and noisy text

### Level 1: Rule-Based Filter

**Tasks:**
1. Download datasets to `data/` directory:
   - [GameTox dataset](https://github.com/shucoll/GameTox) - labeled gaming chat messages
   - [Reddit Usernames dataset](https://www.kaggle.com/datasets/colinmorris/reddit-usernames) - real usernames for testing. Note that these are not labeled as offensive or not

2. Start simple - single word detection:
   - Write a script that counts what percentage of GameTox messages contain "damn"
   - Manually review some of those messages - are they all actually toxic?
   - Count how many messages your script flags (all messages with "damn")
   - Count how many of those are actually labeled as toxic in GameTox (correct flags)
   - Count how many are labeled as not toxic (incorrect flags)

3. Build a regex-based profanity detector:
   - Start with a small list of profane words (5-10 words)
   - Create a regular expression that matches any of those words
   - Test it and observe the same counts as above: total flagged, correct flags, incorrect flags
   - Expand your word list (download a profanity list or grow your hand-written one)
   - Binary classification: profane vs. clean (ignore GameTox's multiple categories for now)

4. Formal evaluation:
   - Calculate accuracy, precision, and recall on GameTox data
   - Review flagged Reddit usernames to find false positives
   - Try to bypass your own filter with creative misspellings - what gets through?

5. Baseline comparison:
   - Compare against [alt-profanity-check](https://github.com/dimitrismistriotis/alt-profanity-check)

**Definition of Done:** When you observe the classic precision/recall tradeoff (blocking offensive content inevitably catches some innocent text) you're ready to move on.

**Key Learnings:**
- Regular expressions
- Working with real-world datasets (incomplete documentation, inherent biases)
- Evaluation metrics (accuracy, precision) vs. traditional testing approaches

**Terminology**
- **Accuracy**: The percentage of predictions that are correct (both positive and negative). Can be misleading with imbalanced datasets.
- **Precision**: Of all items flagged as positive, what percentage are actually positive? High precision means few false alarms.
- **Recall**: Of all actual positive items, what percentage did we catch? High recall means we don't miss much.
- **False positive**: An item incorrectly classified as positive (e.g., flagging "assessment" as profane).
- **False negative**: An item incorrectly classified as negative (e.g., missing an actual profane message).
- **Regular expression**: A pattern-matching language for finding text sequences (e.g., `\b(bad|worse|worst)\b` matches those exact words).
- **Binary classification**: A task with exactly two possible outcomes (e.g., profane vs. clean).


### Level 2: LLM-Based Filter

**Important: Rate Limits for Free Models**

OpenRouter has strict rate limits for free models:
- **50 requests per day** if you have less than $10 in credits
- **1000 requests per day** if you purchase at least $10 in credits
- **20 requests per minute** velocity limit

**Recommended approach**: Start small with batches of 10 messages to stay well under the daily limit. Even with 50-100 messages, you can meaningfully compare LLM vs regex performance. Hard examples (messages where regex struggles) are especially valuable for demonstrating differences between approaches.

See [OpenRouter rate limits documentation](https://openrouter.ai/docs/api/reference/limits) for details.

**Tasks:**
1. Implement an LLM-powered profanity detector:
   - Use [OpenRouter](https://openrouter.ai/) for easy access to free models
   - Design an effective prompt for binary classification
   - **Start with small samples** (10 messages) due to rate limits

2. Model comparison:
   - Test multiple models (e.g., `openai/gpt-oss-20b:free`, `x-ai/grok-4.1-fast`, `meta-llama/llama-3.3-70b-instruct:free`)
   - Compare accuracy, speed, and behavior across models

3. Production feasibility:
   - Calculate costs for processing 1M messages/day with a paid model (e.g., `openai/gpt-5.1`)
   - Compare performance against your Level 1 solution

**Extra Credit:**
- Optimize prompts through systematic experimentation
- Implement response caching to reduce average latency and cost
- Use structured/JSON output mode for reliable parsing
- Extend to multi-class classification (clean / profanity / insult / hate speech)

**Key Learnings:**
- LLM API integration and prompt engineering
- Cost/latency tradeoffs in production ML systems
- When LLMs are (and aren't) practical solutions

**Terminology**
- **Prompt engineering**: The process of designing and refining text instructions to get better results from LLMs. Small wording changes can significantly impact accuracy.
- **Structured output**: Requesting LLMs to return responses in a specific format like JSON, making parsing more reliable than free-form text.
- **Multi-class**: Classification with more than two categories (e.g., clean / profanity / insult / hate speech).
- **Latency**: The time delay between sending a request and receiving a response. Critical for real-time applications like chat filtering.

### Level 3: Traditional ML Classifier

**Tasks:**
1. Train a scikit-learn text classifier:
   - Split GameTox data into training and test sets
   - Choose appropriate features. Start with TfidfVectorizer(ngram_range=(1, 1))
   - Train a classifier. Start with LogisticRegression

2. Comprehensive evaluation:
   - Measure precision, recall, and F1-score on held-out test data
   - Compare against Levels 1 and 2 across multiple dimensions:
     - Accuracy and other metrics
     - Latency (inference speed)
     - Memory footprint
     - External dependencies and costs

**Helpful Resources:**
- [scikit-learn text classification tutorial](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html)
- [GeeksforGeeks NLP classification guide](https://www.geeksforgeeks.org/nlp/text-classification-using-scikit-learn-in-nlp/)

**Extra Credit:**
- Package your model in a scikit-learn Pipeline for cleaner deployment
- Experiment with hyperparameters:
    - TfidfVectorizer: analyzer=word vs char, ngram_range=(1, 2), min_df=1, 2, 3, and so on
    - LogisticRegression: Try different C values

**Key Learnings:**
- Train/test methodology and avoiding overfitting
- Precision, recall, F1-score, and when to optimize for each
- Traditional ML as a practical middle ground between rules and LLMs

**Terminology**
- **Training vs testing data**: Training data is used to build the model; testing data evaluates how well it generalizes to unseen examples.
- **Held-out data (same as testing data)**: Data deliberately set aside and not used during training, reserved exclusively for evaluation.
- **F1-score, F-score, F-measure**: The harmonic mean of precision and recall, balancing both metrics. Commonly used instead of accuracy when the output is rare (most messages don't have profanity).
- **Document frequency**: How many documents contain a particular word. Rare words (low DF) are often more informative than common ones.
- **TF-IDF**: Term Frequency-Inverse Document Frequency; weights words by how often they appear in a document relative to how rare they are overall. Reduces impact of common words like "the".
- **Stop words**: Extremely common words ("the", "a", "is") that are often removed because they add little meaning. May or may not help depending on your task.
- **Ngram**: A sequence of N consecutive words (or characters). Bigrams (2-grams) like "very bad" can capture meaning that single words miss.
- **Logistic regression**: A simple but effective classification algorithm that learns weights for features and outputs a probability. Despite its name, it's used for classification, not regression.

### Level 4: Advanced Directions

Choose one or more extensions based on your interests:

**ML/AI Approaches:**
- Fine-tune a transformer model like [ModernBERT](https://huggingface.co/blog/modernbert) on your dataset
- Benchmark against pre-trained models like [toxic-bert](https://huggingface.co/unitary/toxic-bert)
- Fine-tune an LLM to match GPT-4 performance at lower cost/latency
- Explore censoring (****ing) via token-level approaches (BERT) vs. generative approaches (T5)
- Distinguish between profanity, hate speech, and harassment (should these be handled differently in the game?)

**Expanded Support:**
- Extend to non-English languages using multilingual BERT or LLMs
- **Important**: Validate data quality carefully for languages you don't speak
- Evaluate on additional datasets (see `docs/Perplexity_note.md` for options)

**Deployment & Engineering:**
- Package your solution as an installable Python library
- Build a web API for real-time filtering

**Terminology**
- **Fine-tuning**: Taking a pre-trained model and continuing to train it on your specific dataset. Leverages existing knowledge while specializing to your task.
- **Transformer**: A neural network architecture using attention mechanisms to process sequences. The foundation for BERT, GPT, and most modern LLMs.
- **Generative**: Models that produce new text rather than just classifying existing text. Can "rewrite" profane messages into clean versions rather than just detecting them.