# Profanity Filter Project - Master Status

**Last Updated:** 2025-12-02
**Deadline:** Saturday, 2025-12-07 (5 days remaining)
**Current Token Budget:** 153K remaining (recommend fresh 200K session)

---

## Quick Start for Fresh Session

**To continue, read this file, then execute:**
```bash
cd level4-advanced
source .venv/bin/activate
# Start with additional datasets evaluation
```

**Next immediate action:** Additional datasets evaluation (see Level 4 section below)

---

## Project Overview

Educational project building profanity filters for gaming chat using 4 different approaches. Each level is a **complete, standalone solution** - not sequential steps. Goal is to compare tradeoffs (accuracy, speed, cost, complexity).

**Dataset:** GameTox (53,701 gaming chat messages, 81% clean / 19% toxic)
**Test Set:** 200-message stratified subset (consistent across all levels)

---

## Overall Progress

| Level | Approach              | Status | F1    | Precision | Recall | Latency   | Cost/1M   |
|-------|-----------------------|--------|-------|-----------|--------|-----------|-----------|
| 1     | Rule-Based            | ✅ DONE | 0.650 | 58.0%     | 74.0%  | <1ms      | $0        |
| 2     | LLM (OpenRouter)      | ✅ DONE | 0.816 | 76.9%     | 87.0%  | 181ms     | $1,710    |
| 3     | Traditional ML        | ✅ DONE | 0.677 | 84.0%     | 56.8%  | 0.04ms    | $0        |
| 4     | Toxic-BERT            | ✅ DONE | 0.635 | 76.9%     | 54.1%  | 7.95ms    | $0        |
| 4     | ModernBERT (planned)  | 🔄 TODO | 0.75-0.85 (est) | TBD | TBD | TBD | $0 |
| 4     | Ensemble (planned)    | 🔄 TODO | TBD   | TBD       | TBD    | TBD       | $0        |
| 4     | Multi-class (planned) | 🔄 TODO | -     | -         | -      | -         | -         |
| 4     | Package Library       | 🔄 TODO | -     | -         | -      | -         | -         |

**Current Winner:** Level 2 LLM (F1=0.816) but expensive
**Best Free Option:** Level 3 Traditional ML (F1=0.677, blazing fast, high precision)

---

## Level 1: Rule-Based ✅ COMPLETE

**Approach:** Regex patterns + word lists
**Status:** Complete and documented

**Results:**
- F1: 0.650
- High recall (74%) - catches most toxic messages
- Lower precision (58%) - some false positives
- Instant latency (<1ms)

**Files:**
- `level1-rule-based/` - Complete implementation
- Documentation in README.md

**Key Insight:** Simple and fast, but misses context and has false positives (Scunthorpe problem)

---

## Level 2: LLM-Based ✅ COMPLETE

**Approach:** OpenRouter API with GPT-4-class models
**Status:** Complete and documented

**Results:**
- F1: 0.816 - **BEST ACCURACY**
- High precision (76.9%) and recall (87.0%)
- Understands context and nuance
- 181ms latency (acceptable for async)
- $1,710 per million messages (expensive!)

**Files:**
- `level2-llm-based/` - Complete implementation
- Multiple test runs with different prompts

**Key Insight:** Best accuracy but expensive. Only choose if budget allows and accuracy is critical.

---

## Level 3: Traditional ML ✅ COMPLETE

**Approach:** TF-IDF + Logistic Regression (scikit-learn)
**Status:** Complete and documented

**Results:**
- F1: 0.677 - **BEST FREE OPTION**
- Highest precision (84.0%) - very few false positives
- Lower recall (56.8%) - misses ~43% of toxic messages
- 0.04ms latency - **200x faster than toxic-bert**
- $0 cost after training
- ~500KB model size

**Files:**
- `level3-traditional-ml/train_classifier.py` - Training script
- `level3-traditional-ml/evaluate_classifier.py` - Evaluation
- `level3-traditional-ml/profanity_classifier.pkl` - Trained model
- `level3-traditional-ml/LEVEL3-SUMMARY.md` - Full documentation

**Key Insight:** Surprisingly effective! Beats toxic-bert despite being simpler. Fast, cheap, interpretable.

---

## Level 4: Advanced Directions 🔄 IN PROGRESS

**Strategic Decision:**
1. ✅ Do ALL ML/AI extensions for binary classification
2. ✅ Compare all approaches comprehensively
3. ✅ Pick the best binary classifier
4. ✅ THEN extend winner to multi-class (6 categories)
5. ✅ THEN package as library

### 4a. Toxic-BERT Transformer ✅ COMPLETE

**Approach:** Pre-trained HuggingFace transformer (unitary/toxic-bert)
**Status:** Complete and documented

**Results:**
- F1: 0.635 - **WORSE than Traditional ML!**
- Precision: 76.9% (matches LLM)
- Recall: 54.1% (conservative)
- 7.95ms latency (CPU) → 0.8-4ms with MPS (Apple Silicon GPU)
- $0 cost (runs locally)
- ~440MB model (880x larger than traditional ML)

**Files:**
- `level4-advanced/benchmark_toxicbert.py` - Evaluation script
- `level4-advanced/toxicbert_benchmark.ipynb` - Interactive notebook
- `level4-advanced/LEVEL4-TOXICBERT-SUMMARY.md` - Full documentation
- `level4-advanced/LEVEL4-CONTINUATION-PLAN.md` - Next steps plan

**Key Insight:** Pre-trained transformers don't always beat well-tuned traditional ML! Traditional ML is faster (200x), smaller (880x), and more accurate (F1: 0.677 vs 0.635).

**Technical Note:** Fixed multi-label classification bug - toxic-bert outputs 6 independent scores (sigmoid), not binary (softmax). Take max score across all labels.

---

## Level 4: Remaining Work 🔄 TODO

**Deadline:** Saturday, 2025-12-07
**Token Budget:** 153K remaining (tight) - **RECOMMEND:** restart fresh with 200K

### Next Tasks (Ordered by Priority)

#### 1. Additional Datasets Evaluation (NEXT)

**Goal:** Test if models generalize beyond GameTox

**Approach:**
- Evaluate all 4 approaches on external datasets:
  - Jigsaw Toxic Comments (Wikipedia)
  - Civil Comments (news articles)
  - HateXplain (hate speech)
- Compare F1 scores across datasets
- Identify overfitting vs generalization

**Cost:**
- Time: 2-3 hours
- Tokens: 15-20K
- Remaining after: 133-138K tokens

**Decision Point:**
- If Traditional ML generalizes well → declare winner, skip ModernBERT
- If Traditional ML overfits → proceed with ensemble/ModernBERT

---

#### 2. Ensemble Methods

**Goal:** Combine multiple models for better accuracy

**Approach:**
- Voting: 2+ models agree → toxic
- Weighted: Weight by precision/recall
- Stacking: Meta-classifier on outputs
- Test combinations: Rules + Traditional ML + Toxic-BERT

**Expected:** F1 improvement of 0.02-0.05 (modest)

**Cost:**
- Time: 2-4 hours
- Tokens: 15-25K
- Remaining after: 108-128K tokens

**Decision Point:**
- If ensemble reaches F1 > 0.75 → good enough, proceed to multi-class
- If still insufficient → proceed to ModernBERT

---

#### 3. Censoring/Replacement (Optional)

**Goal:** Replace profanity with asterisks or alternatives

**Approach:**
- Pattern-based (BERT): Mask profane tokens, predict alternatives
- Generative (T5): "Rephrase without profanity"

**Expected:** User-facing enhancement (not accuracy improvement)

**Cost:**
- Time: 1-2 hours
- Tokens: 10-15K
- Remaining after: 93-118K tokens

**Note:** Do AFTER picking best detector, as optional enhancement

---

#### 4. Fine-Tune ModernBERT (HIGH COST)

**Goal:** Custom transformer trained on GameTox

**Approach:**
- Train ModernBERT on full GameTox dataset
- Gaming-specific context understanding
- Expected F1: 0.75-0.85 (between traditional ML and LLM)

**Cost:**
- Time: 3-8 hours (mostly passive - can work on other things)
  - Active: 45-90 min (setup, check-ins, evaluation)
  - Passive: 3-7 hours (training runs in background with `nohup`)
- Tokens: 45-75K
- Remaining after: 78-108K tokens (TIGHT!) or 125-155K if fresh session
- Compute: GPU strongly recommended (MPS on Mac: 3-5 hours)

**Decision Point:**
- ONLY do this if cheaper options prove insufficient
- Requires fresh 200K token session for comfort

---

#### 5. Multi-Class Extension

**Goal:** Extend best binary classifier to 6 categories

**Categories:**
1. Clean
2. Profanity (mild cursing)
3. Insult (personal attacks)
4. Hate speech (identity-based)
5. Threat (violence)
6. Severe toxic (extreme)

**Approach:**
- Take winning binary classifier
- Retrain/adapt for multi-class (softmax output)
- Evaluate on GameTox full labels

**Expected:** More nuanced filtering (e.g., allow mild cursing, block hate speech)

**Cost:**
- Time: 2-4 hours
- Tokens: 20-30K
- Remaining after: TBD (depends on previous steps)

---

#### 6. Package as Library

**Goal:** Production-ready Python library

**Features:**
- Simple API: `filter.classify(message) → {toxic: bool, confidence: float}`
- Multi-class support: `filter.classify_detailed(message) → {category: str, scores: dict}`
- CLI tool: `profanity-filter --file messages.txt`
- Installable: `pip install gaming-profanity-filter`

**Tasks:**
- Create package structure
- Write setup.py / pyproject.toml
- Add tests (pytest)
- Write documentation
- Publish to PyPI (optional)

**Cost:**
- Time: 3-5 hours
- Tokens: 20-30K
- Remaining after: TBD

---

## Token Budget Summary

**Current Session:** 153K remaining

**Estimated Costs:**
- Additional datasets: 15-20K → 133-138K remaining
- Ensemble: 15-25K → 108-128K remaining
- Censoring: 10-15K → 93-118K remaining
- ModernBERT: 45-75K → 78-108K remaining (TIGHT!)
- Multi-class: 20-30K → 58-88K remaining
- Packaging: 20-30K → 28-68K remaining

**Total if doing everything:** 125-195K tokens needed

**Recommendation:** **Restart with fresh 200K session** for comfortable buffer

---

## Key Insights

### Surprising Results

1. **Traditional ML (Level 3) beats Toxic-BERT (Level 4)**
   - F1: 0.677 vs 0.635
   - 200x faster (0.04ms vs 7.95ms)
   - 880x smaller (500KB vs 440MB)
   - Higher precision (84% vs 77%)

2. **Pre-trained transformers ≠ always better**
   - Toxic-bert trained on general web comments
   - Traditional ML trained on gaming-specific data (GameTox)
   - Domain-specific training matters more than model complexity

3. **LLM accuracy comes at a cost**
   - Best F1 (0.816) but $1,710 per million messages
   - Traditional ML is 27% less accurate but FREE and 4,525x faster

### Production Recommendations

**For gaming chat (1M messages/day):**
- **If budget allows:** Level 2 LLM (best accuracy, handles nuance)
- **If budget-constrained:** Level 3 Traditional ML (good accuracy, instant, free)
- **If maximum speed needed:** Level 1 Rules (acceptable accuracy, <1ms)
- **Avoid:** Toxic-BERT (no advantages - slower than traditional ML, less accurate)

**For multi-language support:**
- Consider fine-tuning multilingual-BERT
- Or use LLM with language-specific prompts

---

## File Structure

```
tek_innov8rs_profanity_filter/
├── PROJECT-STATUS.md                    # ← YOU ARE HERE
├── README.md                            # Original instructions
├── CLAUDE.md                            # Claude Code guidance
├── data/                                # Shared datasets (gitignored)
│   ├── GameTox/
│   └── test_subset_200_stratified.csv   # Common test set
│
├── level1-rule-based/                   # ✅ COMPLETE
│   └── (rule-based implementation)
│
├── level2-llm-based/                    # ✅ COMPLETE
│   └── (LLM implementation)
│
├── level3-traditional-ml/               # ✅ COMPLETE
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── profanity_classifier.pkl         # Trained model
│   └── LEVEL3-SUMMARY.md                # Full docs
│
└── level4-advanced/                     # 🔄 IN PROGRESS
    ├── benchmark_toxicbert.py           # ✅ COMPLETE
    ├── toxicbert_benchmark.ipynb        # ✅ COMPLETE
    ├── LEVEL4-TOXICBERT-SUMMARY.md      # ✅ COMPLETE
    ├── LEVEL4-CONTINUATION-PLAN.md      # ✅ COMPLETE
    └── (future: ModernBERT, ensemble, multi-class, package)
```

---

## Immediate Next Steps

**When you restart the fresh session:**

1. **Read this file** (`PROJECT-STATUS.md`)
2. **Navigate to Level 4:**
   ```bash
   cd level4-advanced
   source .venv/bin/activate
   ```
3. **Start Additional Datasets Evaluation:**
   - Download Jigsaw Toxic Comments dataset
   - Create `evaluate_external_datasets.py` script
   - Test all 4 approaches on external data
   - Compare F1 scores and identify winner
4. **Decide next step based on results**

---

## Questions to Resolve

1. **Does Traditional ML (F1=0.677) generalize to other datasets?**
   - If yes → declare winner, proceed to multi-class
   - If no → try ensemble or ModernBERT

2. **Is F1=0.677 good enough for production?**
   - Depends on false negative tolerance (misses 43% toxic messages)
   - May need business context

3. **Should we invest in ModernBERT (45-75K tokens, 3-8 hours)?**
   - Only if cheaper options prove insufficient
   - Expected F1: 0.75-0.85 (not guaranteed to beat LLM)

---

## Success Criteria

**By Saturday deadline, complete:**
- ✅ Level 4 extensions (datasets, ensemble, possibly ModernBERT)
- ✅ Identify best binary classifier with justification
- ✅ Multi-class extension on winner
- ✅ Package as production-ready library
- ✅ Comprehensive documentation

**Deliverable:** Python package that production teams can `pip install` and use immediately.

---

**Status:** Ready to continue
**Recommended:** Exit current session, restart fresh with 200K tokens
**Next Action:** Additional datasets evaluation → decision point → proceed based on results

**Good luck completing by Saturday!** 🚀
