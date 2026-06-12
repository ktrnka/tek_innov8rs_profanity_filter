#!/usr/bin/env python3
"""
Evaluate all 4 profanity filter approaches on external datasets.

Tests generalization beyond GameTox (World of Tanks chat).

Approaches tested:
1. Level 1: Rule-Based (regex + word lists)
2. Level 2: LLM-Based (SKIPPED - too expensive for large-scale testing)
3. Level 3: Traditional ML (TF-IDF + Logistic Regression)
4. Level 4: Toxic-BERT (transformer)

Datasets:
- GameTox (baseline) - World of Tanks chat
- Civil Comments - Wikipedia/news comments
- Real Toxicity Prompts - Web text
- Surge AI - Social media

Goal: Determine if Traditional ML (F1=0.677 on GameTox) generalizes well.
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle

# ============================================================================
# Load Models from Other Levels
# ============================================================================

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

# Level 1: Rule-Based Filter
print("\n[1/3] Loading Level 1: Rule-Based Filter...")
try:
    sys.path.insert(0, '../level1-rule-based')
    from profanity_filter import ProfanityFilter as RuleBasedFilter
    rule_filter = RuleBasedFilter()
    print("✓ Loaded rule-based filter")
except Exception as e:
    print(f"✗ Error loading rule-based: {e}")
    rule_filter = None

# Level 3: Traditional ML
print("\n[2/3] Loading Level 3: Traditional ML Classifier...")
try:
    with open('../level3-traditional-ml/profanity_classifier.pkl', 'rb') as f:
        ml_classifier = pickle.load(f)
    print("✓ Loaded traditional ML classifier")
except Exception as e:
    print(f"✗ Error loading ML classifier: {e}")
    ml_classifier = None

# Level 4: Toxic-BERT
print("\n[3/3] Loading Level 4: Toxic-BERT...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "unitary/toxic-bert"
    toxicbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    toxicbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    toxicbert_model.eval()

    # Use MPS if available (Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    toxicbert_model.to(device)
    print(f"✓ Loaded Toxic-BERT on {device}")
except Exception as e:
    print(f"✗ Error loading Toxic-BERT: {e}")
    toxicbert_model = None
    toxicbert_tokenizer = None
    device = None

# ============================================================================
# Define Prediction Functions
# ============================================================================

def predict_rule_based(texts):
    """Level 1: Rule-based predictions."""
    if rule_filter is None:
        return None
    return np.array([1 if rule_filter.contains_profanity(text) else 0 for text in texts])

def predict_traditional_ml(texts):
    """Level 3: Traditional ML predictions."""
    if ml_classifier is None:
        return None
    return ml_classifier.predict(texts)

def predict_toxicbert(texts, batch_size=32):
    """Level 4: Toxic-BERT predictions."""
    if toxicbert_model is None or toxicbert_tokenizer is None:
        return None

    predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Tokenize
        inputs = toxicbert_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = toxicbert_model(**inputs)
            logits = outputs.logits

            # Toxic-BERT outputs 6 labels (sigmoid) - take max score
            probs = torch.sigmoid(logits)
            max_scores = probs.max(dim=1).values

            # Threshold at 0.5
            batch_preds = (max_scores >= 0.5).cpu().numpy()
            predictions.extend(batch_preds)

    return np.array(predictions)

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_on_dataset(dataset_name, df, approach_name, predict_fn):
    """Evaluate a single approach on a single dataset."""

    print(f"\n  Testing {approach_name}...")

    if predict_fn is None:
        print(f"    ✗ Model not loaded")
        return None

    try:
        # Get texts and labels
        texts = df['text'].tolist()
        y_true = df['is_toxic'].values

        # Predict
        start_time = time.time()
        y_pred = predict_fn(texts)
        elapsed = time.time() - start_time

        if y_pred is None:
            print(f"    ✗ Prediction failed")
            return None

        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Get confusion matrix for additional insights
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate latency per message
        latency_ms = (elapsed / len(texts)) * 1000

        print(f"    ✓ F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"    ✓ Latency: {latency_ms:.2f}ms/msg | Total: {elapsed:.1f}s")
        print(f"    ✓ TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

        return {
            'dataset': dataset_name,
            'approach': approach_name,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'latency_ms': latency_ms,
            'total_time': elapsed,
            'n_samples': len(texts),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# Load Datasets
# ============================================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)

datasets = {}

# GameTox (baseline)
print("\n[1/4] Loading GameTox test set (baseline)...")
try:
    gametox = pd.read_csv('../data/test_subset_200_stratified.csv')
    # Rename message to text for consistency
    gametox = gametox.rename(columns={'message': 'text'})
    # Convert label to binary (label 0.0 = clean, others = toxic)
    gametox['is_toxic'] = (gametox['label'] != 0.0).astype(int)
    datasets['GameTox'] = gametox
    toxic_pct = (gametox['is_toxic'].sum() / len(gametox)) * 100
    print(f"✓ Loaded {len(gametox)} samples ({toxic_pct:.1f}% toxic)")
except Exception as e:
    print(f"✗ Error: {e}")

# Civil Comments
print("\n[2/4] Loading Civil Comments...")
try:
    civil = pd.read_csv('../data/external_datasets/civil_comments_sample.csv')
    datasets['Civil_Comments'] = civil
    toxic_pct = (civil['is_toxic'].sum() / len(civil)) * 100
    print(f"✓ Loaded {len(civil)} samples ({toxic_pct:.1f}% toxic)")
except Exception as e:
    print(f"✗ Error: {e}")

# Real Toxicity Prompts
print("\n[3/4] Loading Real Toxicity Prompts...")
try:
    real_tox = pd.read_csv('../data/external_datasets/real_toxicity_prompts.csv')
    datasets['Real_Toxicity'] = real_tox
    toxic_pct = (real_tox['is_toxic'].sum() / len(real_tox)) * 100
    print(f"✓ Loaded {len(real_tox)} samples ({toxic_pct:.1f}% toxic)")
except Exception as e:
    print(f"✗ Error: {e}")

# Surge AI
print("\n[4/4] Loading Surge AI Toxicity...")
try:
    surge = pd.read_csv('../data/external_datasets/surge_toxicity.csv')
    # Convert string labels to binary if needed
    if surge['is_toxic'].dtype == 'object':
        surge['is_toxic'] = surge['is_toxic'].map({'Toxic': 1, 'Not Toxic': 0, 1: 1, 0: 0})
    datasets['Surge_AI'] = surge
    toxic_pct = (surge['is_toxic'].sum() / len(surge)) * 100
    print(f"✓ Loaded {len(surge)} samples ({toxic_pct:.1f}% toxic)")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Run Evaluations
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING EVALUATIONS")
print("=" * 70)

results = []

for dataset_name, df in datasets.items():
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # Test each approach
    result = evaluate_on_dataset(dataset_name, df, "Rule-Based", predict_rule_based)
    if result:
        results.append(result)

    result = evaluate_on_dataset(dataset_name, df, "Traditional ML", predict_traditional_ml)
    if result:
        results.append(result)

    result = evaluate_on_dataset(dataset_name, df, "Toxic-BERT", predict_toxicbert)
    if result:
        results.append(result)

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df.to_csv('external_dataset_results.csv', index=False)
print(f"\n✓ Saved results to: external_dataset_results.csv")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Pivot table for easier comparison
pivot = results_df.pivot(index='dataset', columns='approach', values='f1')
print("\nF1 Scores by Dataset and Approach:")
print(pivot.to_string())

print("\n" + "-" * 70)
print("Key Question: Does Traditional ML generalize?")
print("-" * 70)

# Compare Traditional ML performance
trad_ml_results = results_df[results_df['approach'] == 'Traditional ML']
if len(trad_ml_results) > 0:
    gametox_f1 = trad_ml_results[trad_ml_results['dataset'] == 'GameTox']['f1'].values
    external_f1s = trad_ml_results[trad_ml_results['dataset'] != 'GameTox']['f1'].values

    if len(gametox_f1) > 0 and len(external_f1s) > 0:
        gametox_f1 = gametox_f1[0]
        avg_external_f1 = external_f1s.mean()
        drop = gametox_f1 - avg_external_f1
        drop_pct = (drop / gametox_f1) * 100

        print(f"\nTraditional ML Performance:")
        print(f"  GameTox (in-domain):  F1 = {gametox_f1:.3f}")
        print(f"  External (avg):       F1 = {avg_external_f1:.3f}")
        print(f"  Performance drop:     {drop:.3f} ({drop_pct:.1f}%)")

        if drop_pct < 20:
            print(f"\n✓ GOOD GENERALIZATION (< 20% drop)")
            print(f"  → Traditional ML works beyond GameTox!")
            print(f"  → Recommendation: Use Traditional ML, skip ModernBERT")
        elif drop_pct < 40:
            print(f"\n⚠ MODERATE GENERALIZATION (20-40% drop)")
            print(f"  → Traditional ML partially generalizes")
            print(f"  → Recommendation: Consider ensemble or ModernBERT if higher accuracy needed")
        else:
            print(f"\n✗ POOR GENERALIZATION (> 40% drop)")
            print(f"  → Traditional ML is GameTox-specific")
            print(f"  → Recommendation: Invest in ModernBERT or ensemble")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
print(f"\nNext steps:")
print(f"1. Review external_dataset_results.csv")
print(f"2. Decide: ModernBERT, ensemble, or proceed to multi-class?")
print(f"3. Update PROJECT-STATUS.md with findings")
