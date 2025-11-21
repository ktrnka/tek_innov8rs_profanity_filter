# Tek Innov8rs: Profanity Filter

## Context
This project is based on real-world work done at Singularity 6 for the game [Palia](https://palia.com/), a multiplayer online game available on PC/Steam, Switch, Xbox, and PlayStation.

### Project Goals
In a production environment, a profanity filter needs to balance multiple objectives:
- **Player safety**: Prevent and reduce harm from hate speech, harassment, and other toxic behavior through automated detection
- **Industry standards**: Follow gaming conventions where profanity is automatically censored
- **Legal compliance**: There are many proposed regulations about online chat safety, particularly for children
- **Platform requirements**: Satisfy certification requirements from Nintendo, Microsoft, and Sony

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

2. Data exploration:
   - Manually review samples from both datasets
   - Write analysis scripts to understand distributions and patterns

3. Build a regex-based profanity detector:
   - Create or download a profanity word list
   - Generate a regular expression to match profane words
   - Binary classification: profane vs. clean (ignore GameTox's multiple categories for now)

4. Evaluation:
   - Test your filter on GameTox data and measure performance
   - Review flagged Reddit usernames for false positives
   - Attempt to bypass your own filter with creative misspellings

5. Baseline comparison:
   - Compare against [alt-profanity-check](https://github.com/dimitrismistriotis/alt-profanity-check)

**Definition of Done:** When you observe the classic precision/recall tradeoff (blocking offensive content inevitably catches some innocent text) you're ready to move on.

**Key Learnings:**
- Regular expressions
- Working with real-world datasets (incomplete documentation, inherent biases)
- Evaluation metrics (accuracy, precision) vs. traditional testing approaches

### Level 2: LLM-Based Filter

**Tasks:**
1. Implement an LLM-powered profanity detector:
   - Use [OpenRouter](https://openrouter.ai/) for easy access to free models
   - Design an effective prompt for binary classification

2. Model comparison:
   - Test multiple models (e.g., `openai/gpt-oss-20b:free`, `x-ai/grok-4.1-fast`, `meta-llama/llama-3.3-70b-instruct:free`)
   - Compare accuracy, speed, and behavior across models

3. Production feasibility:
   - Calculate costs for processing 1M messages/day with a paid model (e.g., `openai/gpt-5.1`)
   - Compare performance against your Level 1 solution

**Extra Credit:**
- Implement response caching to reduce average latency and cost
- Use structured/JSON output mode for reliable parsing
- Optimize prompts through systematic experimentation
- Extend to multi-class classification (clean / profanity / insult / hate speech)

**Key Learnings:**
- LLM API integration and prompt engineering
- Cost/latency tradeoffs in production ML systems
- When LLMs are (and aren't) practical solutions

### Level 3: Traditional ML Classifier

**Tasks:**
1. Train a scikit-learn text classifier:
   - Split GameTox data into training and test sets
   - Choose appropriate features. Start with TfidfVectorizer(ngram_range=(1, 1))
   - Train a classifier. Start with LogisticRegression

2. Comprehensive evaluation:
   - Measure precision, recall, and F1-score on held-out test data
   - Compare against Levels 1 and 2 across multiple dimensions:
     - Accuracy metrics
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
- Build a REST API for real-time filtering
