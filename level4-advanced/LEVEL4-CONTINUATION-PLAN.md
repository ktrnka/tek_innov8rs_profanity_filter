# Level 4 Continuation Plan

**Date Created:** 2025-12-01
**Status:** Ready to continue
**Remaining Tokens:** ~107K (as of last session)

---

## What's Been Completed ✅

### Level 4: Toxic-BERT Transformer (COMPLETE)
- ✅ Created `benchmark_toxicbert.py` - Python script for evaluation
- ✅ Created `toxicbert_benchmark.ipynb` - Jupyter notebook with visualizations
- ✅ Enabled MPS (Apple Silicon GPU) support for 2-10x speedup
- ✅ Fixed multi-label classification (sigmoid, not softmax)
- ✅ Created `LEVEL4-TOXICBERT-SUMMARY.md` - comprehensive documentation
- ✅ Evaluated on 200-message test set

**Results:**
- F1-Score: 0.635 (between rules and traditional ML)
- Precision: 76.9% (matches LLM exactly!)
- Recall: 54.1% (conservative)
- Latency: 7.95ms per message (CPU) → 0.8-4ms with MPS
- Cost: $0 (runs locally)
- Model Size: ~440MB

**Key Finding:** Traditional ML (Level 3) with F1=0.677 actually outperforms toxic-bert's F1=0.635!

---

## Strategic Decision: Progression Order

**Agreed approach (excellent engineering thinking):**

1. **Complete ALL ML/AI extensions for binary classification** (don't pick winner yet)
2. **Compare all binary approaches comprehensively**
3. **Pick the best binary classifier** based on tradeoffs
4. **THEN extend winner to multi-class** (6 categories: clean, profanity, insult, hate, threat, severe)
5. **THEN package as library** (dead last - only package what's proven best)

**Rationale:**
- Don't waste effort on multi-class for inferior models
- Don't package before knowing what's actually best
- Systematic layered evaluation minimizes wasted effort

---

## Next Options: Ordered by Cost

### Current Binary Classification Results

| Approach              | F1    | Precision | Recall | Latency   | Cost/1M   | Complexity |
|-----------------------|-------|-----------|--------|-----------|-----------|------------|
| Level 1: Rules        | 0.650 | 58.0%     | 74.0%  | <1ms      | $0        | Low        |
| Level 2: LLM          | 0.816 | 76.9%     | 87.0%  | 181ms     | $1,710    | Low        |
| Level 3: Trad ML      | 0.677 | 84.0%     | 56.8%  | 0.04ms    | $0        | Medium     |
| Level 4: Toxic-BERT   | 0.635 | 76.9%     | 54.1%  | 7.95ms    | $0        | Medium     |

**Current Winner:** Level 2 LLM (F1=0.816) but expensive
**Best Free Option:** Level 3 Traditional ML (F1=0.677, blazing fast, high precision)

### Option 1: Fine-Tune ModernBERT (MEDIUM-HIGH COST)

**Goal:** Custom transformer trained on GameTox for gaming-specific toxicity detection

**Expected Results:**
- F1-Score: 0.75-0.85 (between traditional ML and LLM)
- Better than toxic-bert because trained on gaming chat specifically
- Likely to beat traditional ML (0.677) but unsure if reaches LLM (0.816)

**Costs:**
- **Time:** 3-8 hours (GPU) or 8-12 hours (CPU only)
  - Active involvement: 45-90 min total (setup, check-ins, evaluation)
  - Passive time: Training runs in background - **can work on other things**
- **Claude Tokens:** 45-75K (script creation, debugging, evaluation, documentation)
  - Would use ~50-70% of remaining 107K tokens
- **Compute:** GPU strongly recommended (MacBook MPS: 3-5 hours, CPU: 8-12 hours)
- **Storage:** ~1-2GB for trained model

**Time Involvement:**
- Setup script: 30-45 min (active)
- Start training: 5-10 min (active)
- Training runs: 3-7 hours (passive - use `nohup`, work on other things)
- Evaluation: 10 min (active)

**Workflow:**
```bash
# Set up training (active: 45 min)
python train_modernbert.py

# Start training in background (active: 5 min)
nohup python train_modernbert.py > training.log 2>&1 &

# Work on other things for 3-7 hours (passive)

# Check results when complete (active: 10 min)
python evaluate_modernbert.py
```

**Pros:**
- Gaming-specific (trained on GameTox slang, leetspeak, gaming insults)
- Likely best accuracy of all free options
- Context understanding better than TF-IDF
- Zero API cost after training

**Cons:**
- High upfront time/token cost
- Large model (~1-2GB)
- Slower inference than traditional ML
- Requires GPU for practical training time

**Recommendation:** ONLY if traditional ML (F1=0.677) proves insufficient for production needs

---

### Option 2: Censoring/Replacement (LOW COST)

**Goal:** Replace detected profanity with asterisks or alternative words

**Approaches:**
- **Pattern-based (BERT):** Mask profane tokens with [MASK], let BERT predict alternatives
- **Generative (T5):** "Rephrase without profanity: {message}" → cleaned version

**Expected Results:**
- Not a detection approach - augments existing detector
- Quality depends on base model
- Useful for user-facing output (show cleaned version)

**Costs:**
- **Time:** 1-2 hours (prototype and test)
- **Claude Tokens:** 10-15K (implementation and examples)
- **Compute:** Similar to toxic-bert (7-10ms per message)

**Pros:**
- Low cost to explore
- Adds user-facing feature (show why flagged, offer cleaned version)
- Works with ANY detector (rules, ML, LLM)

**Cons:**
- Context errors ("That's sick!" → "That's [MASK]" loses positive meaning)
- Not a standalone solution (needs detector first)
- May create awkward rephrasing

**Recommendation:** Do AFTER picking best detector, as optional enhancement

---

### Option 3: Additional Datasets Evaluation (LOW COST)

**Goal:** Test how well models generalize to other toxic comment datasets

**Datasets to try:**
- Jigsaw Toxic Comments (Wikipedia talk pages)
- Civil Comments (news article comments)
- HateXplain (hate speech with explanations)

**Expected Results:**
- Reveals overfitting to GameTox
- Shows which approach generalizes best
- Informs production deployment decisions

**Costs:**
- **Time:** 2-3 hours (download datasets, run evaluations)
- **Claude Tokens:** 15-20K (scripts and analysis)
- **Compute:** Minimal (just inference)

**Pros:**
- Low cost
- Validates production readiness
- May reveal surprising strengths/weaknesses

**Cons:**
- Different domains (Wikipedia ≠ gaming chat)
- May not reflect actual use case
- Results may be discouraging if models are overfit

**Recommendation:** Do BEFORE fine-tuning ModernBERT (might reveal traditional ML is already good enough)

---

### Option 4: Ensemble Methods (LOW COST)

**Goal:** Combine predictions from multiple models for better accuracy

**Approaches:**
- Voting: 2+ models agree → flag as toxic
- Weighted: Weight models by precision/recall
- Stacking: Train meta-classifier on outputs

**Expected Results:**
- F1 improvement of 0.02-0.05 (modest)
- Precision boost (multiple models reduce false positives)
- Recall boost (catch what individual models miss)

**Costs:**
- **Time:** 2-4 hours (implement and evaluate)
- **Claude Tokens:** 15-25K (code and analysis)
- **Compute:** Runs multiple models (latency = sum of individual models)

**Example ensemble:**
- Rules (fast, high recall) + Traditional ML (high precision) + Toxic-BERT (moderate both)
- If ANY model predicts toxic with >80% confidence → flag
- Reduces false negatives while maintaining precision

**Pros:**
- Often beats individual models
- Can combine strengths (rules catch obvious, ML catches subtle)
- No training required

**Cons:**
- Higher latency (runs multiple models)
- More complex deployment
- Gains may be minimal

**Recommendation:** Try AFTER evaluating all individual approaches

---

### Option 5: Fine-Tune LLM to Match GPT-4 (VERY HIGH COST)

**Goal:** Create custom LLM fine-tuned on GameTox to match GPT-4 accuracy at lower cost

**Expected Results:**
- F1-Score: 0.80-0.85 (similar to GPT-4's 0.816)
- Faster inference than GPT-4 API
- Lower cost per message (if self-hosted)

**Costs:**
- **Time:** 5-12 hours (data prep, training, evaluation)
- **Claude Tokens:** 60-100K (would exceed remaining budget!)
- **Compute:** Requires GPU, potentially cloud GPU ($$)
- **API Costs:** If using OpenAI fine-tuning API: $30-100+ for training

**Pros:**
- Potentially matches best accuracy (F1=0.816)
- Lower latency than API calls (if self-hosted)
- Gaming-specific context

**Cons:**
- Extremely high cost (time, tokens, compute, API)
- May not beat GPT-4 (which has orders of magnitude more training)
- Deployment complexity (hosting, scaling)
- Exceeds remaining token budget

**Recommendation:** SKIP - costs outweigh benefits. If LLM-level accuracy needed, use API (Level 2)

---

## Recommended Next Steps

### Immediate (Low Cost, High Value)

1. **Additional Datasets Evaluation** (2-3 hours, 15-20K tokens)
   - Test all 4 current approaches on Jigsaw, Civil Comments
   - Reveals generalization vs overfitting
   - May show traditional ML is already good enough

2. **Ensemble Methods** (2-4 hours, 15-25K tokens)
   - Combine Rules + Traditional ML + Toxic-BERT
   - Likely modest F1 improvement (0.02-0.05)
   - No training required

### Medium Priority (If Time/Tokens Allow)

3. **Censoring Prototype** (1-2 hours, 10-15K tokens)
   - Pattern-based BERT masking
   - User-facing feature enhancement
   - Works with any detector

### High Cost (Only if Needed)

4. **Fine-Tune ModernBERT** (3-8 hours, 45-75K tokens)
   - ONLY if traditional ML proves insufficient
   - Expected F1: 0.75-0.85
   - Requires GPU for practical timing

---

## Decision Points

### After Additional Datasets Evaluation

**If traditional ML (Level 3) generalizes well across datasets:**
- ✅ Declare traditional ML the winner
- ✅ Extend to multi-class (6 categories)
- ✅ Package as library
- ❌ Skip ModernBERT (not worth the cost)

**If traditional ML overfits to GameTox:**
- ✅ Try ensemble methods first (low cost)
- ✅ If still insufficient, pursue ModernBERT
- ❌ Consider if LLM API (Level 2) is more practical

### After Ensemble Methods

**If ensemble reaches F1 > 0.75:**
- ✅ Good enough for production
- ✅ Extend to multi-class
- ✅ Package ensemble as library

**If ensemble still < 0.75 and LLM cost is prohibitive:**
- ✅ Pursue ModernBERT fine-tuning
- ⚠️ Be prepared for 3-8 hour commitment

---

## Token Budget Tracking

**Started Level 4 with:** 162K tokens
**After toxic-bert completion:** ~107K tokens remaining
**Token costs per remaining option:**

| Option                        | Tokens | Remaining After |
|-------------------------------|--------|-----------------|
| Additional Datasets           | 15-20K | 87-92K          |
| Ensemble Methods              | 15-25K | 62-82K          |
| Censoring Prototype           | 10-15K | 52-72K          |
| Fine-Tune ModernBERT          | 45-75K | 12-62K          |
| Fine-Tune LLM (NOT RECOMMENDED)| 60-100K| 7-47K (risky!)  |

**Recommended sequence to maximize value:**
1. Additional Datasets (20K) → 87K remaining
2. Ensemble Methods (25K) → 62K remaining
3. Censoring (15K) → 47K remaining
4. If needed: ModernBERT (50K) → ~0K remaining

**Alternative conservative sequence:**
1. Additional Datasets (20K) → 87K remaining
2. **Decide:** If traditional ML wins, stop here and go to multi-class
3. If not: Ensemble (25K) → 62K remaining
4. **Decide:** If ensemble wins, stop here and go to multi-class
5. If not: ModernBERT (50K) → 12K remaining

---

## Files Created This Session

### Level 4 Directory: `/Users/atman/Innov8tors/tek_innov8rs_profanity_filter/level4-advanced/`

- ✅ `benchmark_toxicbert.py` (269 lines) - Python evaluation script
- ✅ `toxicbert_benchmark.ipynb` (11 sections, 24 cells) - Jupyter notebook
- ✅ `LEVEL4-TOXICBERT-SUMMARY.md` (427 lines) - Comprehensive documentation
- ✅ `pyproject.toml` - UV project config with dependencies
- ✅ `.venv/` - Virtual environment with transformers, torch, pandas, scikit-learn

**Model artifact:** toxic-bert downloads to `~/.cache/huggingface/` (~440MB)

---

## Key Insights to Remember

1. **Traditional ML (Level 3) currently beats toxic-bert** (F1: 0.677 vs 0.635)
   - 200x faster (0.04ms vs 7.95ms)
   - 880x smaller (500KB vs 440MB)
   - Higher precision (84% vs 77%)

2. **LLM (Level 2) still has best accuracy** (F1: 0.816)
   - But costs $1,710 per million messages
   - 181ms latency (not great for real-time)

3. **Multi-label vs multi-class distinction matters:**
   - Toxic-bert is multi-label (6 independent scores with sigmoid)
   - Multi-class would be mutually exclusive categories (softmax)
   - For binary: take max score across all labels

4. **MPS (Apple Silicon GPU) works:**
   - Torch 2.9.1 has MPS support built-in
   - Use `torch.backends.mps.is_available()` to detect
   - Expected 2-10x speedup vs CPU

5. **ModernBERT fine-tuning is mostly passive:**
   - 45-90 min active involvement
   - 3-7 hours passive (can work on other things)
   - Use `nohup` to run in background

---

## Questions to Answer Tomorrow

1. **Which path to take?**
   - Conservative: Additional datasets → decide
   - Thorough: Additional datasets → ensemble → decide
   - Aggressive: Go straight to ModernBERT

2. **Is F1=0.677 (traditional ML) good enough for production?**
   - Depends on cost tolerance
   - Depends on false negative impact (misses 43% of toxic messages)
   - May need business context

3. **Should we optimize for precision or recall?**
   - High precision: Fewer false alarms, better UX, but misses toxic content
   - High recall: Catch more toxic content, but more false positives
   - Current models are precision-heavy (good for UX)

---

## Ready to Continue

**Pick up from here tomorrow:**
1. Review this plan
2. Choose next option (recommend: Additional Datasets)
3. Execute
4. Evaluate results
5. Decide next step based on outcomes

**If short on time/tokens:**
- Skip straight to multi-class on traditional ML (current winner)
- Package traditional ML as library
- Call Level 4 complete

**If want to be thorough:**
- Additional datasets → ensemble → compare all approaches
- Pick winner → multi-class → package
- Maximum learning, maximum comparison

---

**Status:** ✅ Ready to continue
**Next session goal:** Decide which Level 4 extension to pursue
**End state:** Best binary classifier identified, ready for multi-class and packaging
