# Level 4: Toxic-BERT Transformer - Implementation Summary

## Executive Summary

Level 4 implements **transformer-based profanity detection** using a pre-trained toxic-bert model from Hugging Face. This approach demonstrates that modern neural networks can provide competitive accuracy without the cost and latency of LLM APIs.

**Key Results:**
- **F1-Score:** 0.635 (between Level 1 rules and Level 3 traditional ML)
- **Precision:** 76.9% (matches Level 2 LLM exactly!)
- **Recall:** 54.1% (conservative - prioritizes avoiding false positives)
- **Latency:** 7.95ms per message (200x faster than LLM, but 200x slower than traditional ML)
- **Cost:** $0 after download (runs 100% locally)
- **Model Size:** ~440MB (large, but manageable)

## Approach

### Model Selection: toxic-bert

**What is toxic-bert?**
- Pre-trained BERT transformer fine-tuned for toxicity detection
- Created by Unitary AI
- Trained on Jigsaw Toxic Comments and Civil Comments datasets
- Available on HuggingFace: `unitary/toxic-bert`

**Important:** toxic-bert is a **multi-label** classifier, not binary! It outputs scores for 6 toxicity types:
1. toxic
2. severe_toxic
3. obscene
4. threat
5. insult
6. identity_hate

### Adapting Multi-Label to Binary Classification

For our binary clean/toxic task, we use this strategy:

```python
# toxic-bert outputs 6 scores (one per toxicity type)
logits = model(input_text)  # Shape: [batch_size, 6]

# Use sigmoid (not softmax) for multi-label
probs = torch.sigmoid(logits)  # Each score is independent

# Take maximum score across all 6 types
max_toxicity_score = torch.max(probs, dim=1)

# Classify as toxic if ANY type > threshold (0.5)
is_toxic = max_toxicity_score > 0.5
```

**Why this works:**
- If a message is obscene (score 0.8) but not a threat (score 0.1), it's still toxic
- We care if the message violates ANY toxicity category
- This is more lenient than requiring ALL categories to agree

### Test Dataset

**Same 200-message stratified subset** used in Levels 2 and 3:
- Clean: 163 messages (81.5%)
- Toxic: 37 messages (18.5%)
- Enables direct comparison across all levels

## Test Set Performance

### Metrics

| Metric       | Value | Percentage |
|--------------|-------|------------|
| Accuracy     | 0.885 | 88.5%      |
| Precision    | 0.769 | 76.9%      |
| Recall       | 0.541 | 54.1%      |
| F1-Score     | 0.635 | 63.5%      |

### Confusion Matrix

```
                  Predicted
                  Clean  Toxic
Actual  Clean      157     6
        Toxic       17    20
```

**Breakdown:**
- **True Negatives (TN): 157** - Correctly identified clean messages
- **False Positives (FP): 6** - Clean messages incorrectly flagged as toxic
- **False Negatives (FN): 17** - Toxic messages that were missed
- **True Positives (TP): 20** - Correctly identified toxic messages

### Error Analysis

**False Positive Rate:** 3.7% (6 out of 163 clean messages)
- Very low! Only 6 false alarms out of 163 clean messages
- Good user experience - minimal frustration from incorrect flags

**False Negative Rate:** 45.9% (17 out of 37 toxic messages)
- Conservative behavior - misses nearly half of toxic content
- Similar to Level 3 traditional ML (43.2% FNR)
- Tradeoff: prioritize precision over recall

### Detailed Classification Report

```
              precision    recall  f1-score   support

       Clean      0.902     0.963     0.932       163
       Toxic      0.769     0.541     0.635        37

    accuracy                          0.885       200
   macro avg      0.836     0.752     0.783       200
weighted avg      0.878     0.885     0.877       200
```

### Performance Characteristics

**Latency:**
- Total inference time: 1,589ms for 200 messages
- Per-message latency: **7.95ms**
- Breakdown:
  - Model loading: 0.58s (one-time cost)
  - Inference: 1.59s for 200 messages

**Speed Comparison:**
- 200x slower than Level 3 traditional ML (0.04ms)
- 23x faster than Level 2 LLM (181ms)
- Middle ground for real-time applications

**Hardware:**
- Tested on: CPU only
- Would be significantly faster on GPU (5-10x speedup expected)
- Model runs locally - no API calls or internet required

## Example Predictions

### Correctly Identified Toxic Messages

```
✓ "fucking focus" → TOXIC (99.2% confident)
✓ "*** you strv" → TOXIC (87.5% confident)
```

The model confidently catches explicit profanity, even when partially censored.

### Missed Toxic Messages (False Negatives)

```
✗ "jap" → Predicted CLEAN (99.8% confident)
   True label: TOXIC
```

**Why missed?**
- "jap" is a racial slur (short for "Japanese")
- Likely not in the model's training data (Jigsaw/Civil Comments)
- Gaming-specific or less common slurs may be missed
- Demonstrates limitation of pre-trained models on domain-specific data

### Correctly Identified Clean Messages

```
✓ "gg;)" → CLEAN (99.9% confident)
✓ "going on crossfire" → CLEAN (98.5% confident)
✓ "please" → CLEAN (99.9% confident)
✓ "gg so far" → CLEAN (95.2% confident)
```

The model is very confident about benign gaming chat.

## Cross-Level Comparison

| Metric            | Level 1 Rules | Level 2 LLM | Level 3 Trad ML | Level 4 Toxic-BERT |
|-------------------|---------------|-------------|-----------------|-------------------|
| F1-Score          | 0.650         | 0.816       | 0.677           | 0.635             |
| Precision         | 0.580         | 0.769       | 0.840           | 0.769             |
| Recall            | 0.740         | 0.870       | 0.568           | 0.541             |
| Latency (ms/msg)  | <1            | 181         | 0.04            | 7.95              |
| Cost per 1M msgs  | $0            | $1,710      | $0              | $0                |
| Model size        | <1KB          | N/A (API)   | ~500KB          | ~440MB            |

### Key Insights

**Toxic-BERT's Position:**
1. **Accuracy:** F1 of 0.635 is between rules (0.650) and traditional ML (0.677)
   - Surprisingly, NOT better than traditional ML!
   - Traditional ML (scikit-learn) performs better on this specific dataset

2. **Precision:** 76.9% matches Level 2 LLM exactly
   - Same precision as GPT-4 class models
   - Better than rules (58%), slightly worse than traditional ML (84%)

3. **Recall:** 54.1% is conservative like Level 3
   - Misses nearly half of toxic messages
   - Better than missing 43% (Level 3), but worse than LLM (87%)

4. **Speed:** 7.95ms is a middle ground
   - 200x slower than traditional ML (too slow?)
   - 23x faster than LLM (good!)
   - Still fast enough for real-time chat (< 10ms is acceptable)

5. **Cost:** $0 like rules and traditional ML
   - Huge advantage over LLM's $1,710 per million messages
   - One-time download, then 100% local

6. **Model Size:** 440MB is large
   - 880x bigger than traditional ML (500KB)
   - Requires ~1GB RAM when loaded
   - May be prohibitive for embedded systems

## When to Use Toxic-BERT

### Good Use Cases

1. **Budget constraints + decent accuracy needed**
   - Zero API costs after download
   - Better than rules, competitive with traditional ML
   - Acceptable latency (<10ms)

2. **No labeled training data available**
   - Pre-trained on existing toxicity datasets
   - Works out-of-the-box without fine-tuning
   - Unlike Level 3 which requires GameTox for training

3. **GPU available**
   - CPU inference at 8ms is acceptable
   - GPU would reduce to 1-2ms (competitive with traditional ML)
   - Cost-effective if hardware is available

4. **Multi-language support needed** (with multilingual-BERT)
   - Toxic-bert is English-only
   - But multilingual transformer models exist
   - Traditional ML would need separate training per language

### Not Recommended When

1. **Maximum accuracy required**
   - Level 2 LLM achieves 82% F1 (vs 64%)
   - Traditional ML achieves 68% F1 with 84% precision
   - Toxic-BERT doesn't outperform either

2. **Latency must be <1ms**
   - Traditional ML is 200x faster
   - Rules are even faster
   - Toxic-BERT's 8ms may be too slow for some applications

3. **Resource-constrained environments**
   - 440MB model + 1GB RAM is heavy
   - Traditional ML uses <10MB total
   - Rules use <1MB

4. **Domain-specific toxicity**
   - Missed "jap" (gaming-specific slur)
   - Pre-trained on general web comments
   - Fine-tuning on GameTox might help (but then Level 3 is simpler)

## Technical Challenges and Solutions

### Challenge 1: Model is Multi-Label, Not Binary

**Problem:** toxic-bert outputs 6 toxicity scores, not clean/toxic binary classification

**Solution:**
- Use sigmoid activation (not softmax) for independent scores
- Take maximum score across all 6 types
- Threshold at 0.5: if ANY type > 0.5, classify as toxic
- This captures messages that are toxic in any way

### Challenge 2: Initial 0% Precision/Recall

**Root Cause:** First implementation used softmax + argmax (treating multi-label as multi-class)

**Debugging Steps:**
1. Checked model configuration: saw 6 labels, not 2
2. Realized it's multi-label (multiple can be true) not multi-class (only one is true)
3. Switched from softmax to sigmoid
4. Changed decision rule from argmax to max-then-threshold

**Lesson:** Always check model architecture before assuming classification type!

### Challenge 3: Model Loading Time

**Issue:** 45s initial load time (HuggingFace cache miss)

**Solution:**
- Subsequent loads: 0.58s (cached locally)
- Production deployment: pre-download model to disk
- Use `from_pretrained(local_path)` to skip download

## Implementation Details

### Files Created

1. **benchmark_toxicbert.py** (269 lines)
   - Loads toxic-bert from HuggingFace
   - Runs inference on 200-message test set
   - Calculates all metrics and confusion matrix
   - Generates cross-level comparison

### Dependencies

```toml
[project]
name = "level4-advanced"
requires-python = ">=3.13"
dependencies = [
    "transformers>=4.57.3",
    "torch>=2.9.1",
    "pandas>=2.3.3",
    "scikit-learn>=1.7.2",
]
```

### Usage

**Running the benchmark:**
```bash
cd level4-advanced
source .venv/bin/activate
python benchmark_toxicbert.py
```

**Programmatic use:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model (one-time, cache locally)
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

# Predict toxicity
message = "Your message here"
inputs = tokenizer(message, return_tensors="pt")
outputs = model(**inputs)

# Multi-label classification: use sigmoid
probs = torch.sigmoid(outputs.logits)

# Max score across all 6 toxicity types
max_score = torch.max(probs).item()

# Decision
is_toxic = max_score > 0.5
print(f"Toxic: {is_toxic}, Confidence: {max_score:.2%}")
```

## Conclusions

### What We Learned

1. **Pre-trained transformers are accessible**
   - No training required - download and use
   - Competitive with traditional ML (F1: 63.5% vs 67.7%)
   - Much cheaper than LLM APIs ($0 vs $1,710 per million)

2. **Traditional ML can outperform small transformers**
   - Level 3's scikit-learn (68% F1, 84% precision) beats toxic-bert (64% F1, 77% precision)
   - TF-IDF + LogisticRegression is surprisingly effective
   - 200x faster inference (0.04ms vs 8ms)

3. **Multi-label vs multi-class matters**
   - toxic-bert has 6 independent labels, not 1 categorical label
   - Must use sigmoid (not softmax) and threshold (not argmax)
   - Understanding model architecture is critical

4. **Domain-specific data helps**
   - toxic-bert trained on web comments
   - Missed gaming-specific slur "jap"
   - Fine-tuning on GameTox could improve (but adds complexity)

### Recommendations

**Choose Toxic-BERT when:**
- No labeled training data (can't do Level 3)
- LLM cost is prohibitive (can't do Level 2)
- Need better accuracy than rules (Level 1)
- Have GPU for fast inference
- 64% F1 is acceptable for your use case

**Don't choose Toxic-BERT if:**
- You have labeled data → Use Level 3 traditional ML (better F1, 200x faster, 880x smaller)
- Budget allows → Use Level 2 LLM (82% F1, best accuracy)
- Need <1ms latency → Use Level 1 rules (instant)
- Resource-constrained → Use Level 3 (500KB vs 440MB)

### Future Improvements

**For Toxic-BERT Approach:**
1. **Threshold tuning:** Try 0.3 or 0.7 instead of 0.5 to optimize precision/recall tradeoff
2. **Ensemble with Level 3:** Combine transformer + traditional ML predictions
3. **Fine-tune on GameTox:** Adapt pre-trained model to gaming chat specifically
4. **Try other models:**
   - `martin-ha/toxic-comment-model` (alternative toxic-bert)
   - `unitary/unbiased-toxic-roberta` (reduces bias)
   - `microsoft/deberta-v3-base` (fine-tune yourself)

**Bridging to Advanced Level 4:**
- **ModernBERT fine-tuning:** Train from scratch on GameTox (expect 75-80% F1)
- **Distillation:** Compress toxic-bert to smaller model (reduce 440MB → 50MB)
- **Multi-label output:** Return all 6 toxicity scores for fine-grained filtering
- **Multi-language:** Use XLM-RoBERTa for cross-language support

## Final Thoughts

Toxic-BERT demonstrates an interesting middle ground:
- **Cheaper than LLMs:** $0 vs $1,710 per million messages
- **Easier than fine-tuning:** No training required
- **But:** Doesn't outperform simpler approaches on this dataset

**The surprise winner:** Traditional ML (Level 3) offers better accuracy (68% vs 64% F1), 200x faster speed (0.04ms vs 8ms), and 880x smaller size (500KB vs 440MB).

**The lesson:** Always benchmark simple baselines! A well-tuned TF-IDF + LogisticRegression can compete with (or beat) pre-trained transformers, especially on domain-specific tasks.

**When transformers shine:**
- No labeled data (can't train traditional ML)
- Need transfer learning from related tasks
- Want to leverage massive pre-training (billions of parameters)
- Multi-language or multi-task requirements

For this specific profanity detection task, **Level 3 traditional ML** appears to be the sweet spot: good accuracy, tiny size, blazing fast, zero cost.

---

**Document generated:** 2025-12-01
**Model:** unitary/toxic-bert
**Test dataset:** Stratified 200-message subset
**F1-Score:** 0.635
**Key Finding:** Pre-trained transformers accessible but traditional ML still competitive
**Status:** ✅ Complete
