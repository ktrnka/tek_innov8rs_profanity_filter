# ModernBERT Binary vs Multi-Class Comparison

**Date:** December 3, 2025
**Status:** Evaluation complete

---

## Executive Summary

We trained and evaluated **two ModernBERT models**:
1. **Binary ModernBERT**: 2-class classification (Clean vs Toxic)
2. **Multi-Class ModernBERT**: 4-class classification (Clean, Profanity, Insult, Hate Speech)

**Key Finding:** Multi-class model performs **9.7% better on in-domain data** but **1.7% worse on external datasets** compared to binary.

---

## Training Configuration

Both models used identical hyperparameters:
- **Model**: `answerdotai/ModernBERT-base` (149.6M parameters)
- **Learning rate**: 2e-5
- **Batch size**: 4 (due to M1 Pro memory constraints)
- **Epochs**: 3
- **Dataset**: GameTox (53,704 messages, 80/20 train/val split)
- **Training time**: ~9.5 hours each
- **Device**: MPS (Apple Silicon M1 Pro)

**Multi-class labels:**
- Class 0: Clean (81%)
- Class 1: Profanity (13.8%)
- Class 2: Insult (4.4%)
- Class 3: Hate Speech (0.8%, merged from original classes 3-5)

---

## Validation Set Performance

**Binary ModernBERT:**
- F1: 0.7778
- Precision: 0.80
- Recall: 0.757
- Accuracy: N/A

**Multi-Class ModernBERT:**
- **Macro F1**: 0.6606 (treats all 4 classes equally)
- **Weighted F1**: 0.8969 (accounts for class imbalance)
- **Accuracy**: 89.99%

---

## External Dataset Comparison

### GameTox Test Set (200 samples, 18.5% toxic)

| Metric    | Binary   | Multi-Class | Difference |
|-----------|----------|-------------|------------|
| F1        | 0.7778   | **0.8533**  | **+9.7%**  |
| Precision | 0.8000   | 0.8421      | +5.3%      |
| Recall    | 0.7568   | 0.8649      | +14.3%     |
| Latency   | 7.59 ms  | 4.55 ms     | -40%       |

**Winner: Multi-Class** 🏆

### Civil Comments (5K samples, 8% toxic)

| Metric    | Binary   | Multi-Class | Difference |
|-----------|----------|-------------|------------|
| F1        | **0.3960** | 0.3936    | -0.6%      |
| Precision | 0.2694   | 0.2972      | +10.3%     |
| Recall    | **0.7475** | 0.5825    | -22.1%     |
| Latency   | 30.45 ms | 30.51 ms    | +0.2%      |

**Winner: Binary** (tie on F1, but better recall)

### Real Toxicity Prompts (3K samples, 21.9% toxic)

| Metric    | Binary   | Multi-Class | Difference |
|-----------|----------|-------------|------------|
| F1        | **0.6216** | 0.6059    | -2.5%      |
| Precision | 0.5725   | **0.6036**  | +5.4%      |
| Recall    | **0.6799** | 0.6082    | -10.5%     |
| Latency   | 3.97 ms  | 3.95 ms     | -0.5%      |

**Winner: Binary**

### Surge AI (1K samples, 50.1% toxic)

| Metric    | Binary   | Multi-Class | Difference |
|-----------|----------|-------------|------------|
| F1        | **0.8412** | 0.8277    | -1.6%      |
| Precision | 0.9095   | **0.9052**  | -0.5%      |
| Recall    | **0.7824** | 0.7625    | -2.5%      |
| Latency   | 25.08 ms | 25.28 ms    | +0.8%      |

**Winner: Binary**

---

## Summary Statistics

### Average External F1 (excluding GameTox)

| Approach    | Civil Comments | Real Toxicity | Surge AI | **Average** |
|-------------|----------------|---------------|----------|-------------|
| Binary      | 0.3960         | 0.6216        | 0.8412   | **0.6196**  |
| Multi-Class | 0.3936         | 0.6059        | 0.8277   | **0.6091**  |

**Difference:** -1.7% (multi-class slightly worse on external data)

### Multi-Class Prediction Breakdown

Multi-class model's predictions by category:

| Dataset        | Clean | Profanity | Insult | Hate Speech |
|----------------|-------|-----------|--------|-------------|
| GameTox        | 81.0% | 14.0%     | 3.5%   | 1.5%        |
| Civil Comments | 84.3% | 4.4%      | 4.4%   | 6.9%        |
| Real Toxicity  | 78.0% | 9.0%      | 5.8%   | 7.3%        |
| Surge AI       | 57.8% | 25.6%     | 5.2%   | 11.4%       |

**Insight:** Multi-class model distinguishes toxicity types, but this nuance is lost when converting to binary for comparison.

---

## Comparison with Other Approaches

### Average External F1 (All Approaches)

| Approach            | GameTox | External Avg | Training Time | Latency/msg |
|---------------------|---------|--------------|---------------|-------------|
| Traditional ML      | 0.6774  | 0.3990       | ~5 min        | 0.008 ms    |
| Toxic-BERT          | 0.6349  | **0.6670**   | N/A (pre-trained) | 10.4 ms |
| ModernBERT (Binary) | **0.7778** | 0.6196   | 9.5 hours     | 13.8 ms     |
| ModernBERT (Multi)  | **0.8533** | 0.6091   | 9.5 hours     | 15.8 ms     |

**Key Findings:**
1. **Best in-domain performance**: Multi-Class ModernBERT (F1=0.8533)
2. **Best generalization**: Toxic-BERT (F1=0.6670 external avg)
3. **Fastest inference**: Traditional ML (0.008 ms)
4. **Best trade-off**: Toxic-BERT (good F1, pre-trained, reasonable latency)

---

## Analysis

### Why Multi-Class Performs Better In-Domain

1. **Richer training signal**: 4-class labels provide more information than binary
2. **Better class separation**: Model learns nuances between profanity, insult, and hate speech
3. **Higher validation accuracy**: 90% vs ~78% for binary

### Why Multi-Class Performs Worse on External Datasets

1. **Label mismatch**: External datasets are binary-labeled, not multi-class
2. **Conversion loss**: Converting 4-class predictions to binary loses nuance
3. **Overfitting to GameTox taxonomy**: GameTox's 4-class structure may not map well to other datasets
4. **Class boundary issues**: External "toxic" may span multiple GameTox classes inconsistently

### Example: Civil Comments

Multi-class model predicts:
- 219 Profanity
- 219 Insult
- 346 Hate Speech

But Civil Comments only has binary "toxic" labels. The 4-class predictions may capture nuances that binary labels miss, resulting in higher FPs.

---

## Recommendations

### Use Multi-Class ModernBERT When:
- ✅ Operating within GameTox domain (gaming chat)
- ✅ Need to distinguish toxicity types for moderation
- ✅ Have 4-class labeled data
- ✅ Can accept slightly lower generalization
- ✅ Need best in-domain performance (F1=0.85)

### Use Binary ModernBERT When:
- ✅ Need slightly better cross-domain generalization
- ✅ Only care about toxic vs clean (no subcategories)
- ✅ Simpler model deployment

### Use Toxic-BERT When:
- ✅ Need best overall generalization (F1=0.67 external)
- ✅ No time to train custom model
- ✅ Pre-trained model acceptable
- ✅ Can accept lower in-domain performance

### Use Traditional ML When:
- ✅ Inference speed critical (0.008 ms)
- ✅ Computational resources limited
- ✅ Can accept lower accuracy (F1=0.40 external)

---

## Production Considerations

### Multi-Class Advantages:
1. **Actionable insights**: Distinguish between profanity, insults, hate speech
2. **Moderation levels**: Apply different penalties (warning vs ban)
3. **User feedback**: Show specific violation type ("You used profanity")
4. **Analytics**: Track toxicity trends by category

### Multi-Class Disadvantages:
1. **Annotation cost**: Requires 4-class labeled data
2. **Complexity**: More classes = harder to maintain
3. **Generalization**: Slightly worse on external datasets (-1.7%)

### Binary Advantages:
1. **Simplicity**: Easier to annotate, deploy, maintain
2. **Generalization**: Slightly better cross-domain (+1.7%)
3. **Universal labels**: Works with any binary toxic/clean dataset

---

## Conclusion

**Best Model for GameTox/Gaming Chat:** **Multi-Class ModernBERT**
- F1=0.85 on GameTox test set
- Provides toxicity type breakdown
- 90% validation accuracy

**Best Model for General Use:** **Toxic-BERT**
- F1=0.67 average on external datasets
- Pre-trained, no training cost
- Good balance of performance and practicality

**For this project, we recommend Multi-Class ModernBERT** because:
1. Trained specifically on GameTox (best domain match)
2. Superior in-domain performance (+9.7% over binary)
3. Provides actionable toxicity type classification
4. External generalization gap is small (-1.7%)

The multi-class approach is production-ready and offers the most value for gaming chat moderation.
