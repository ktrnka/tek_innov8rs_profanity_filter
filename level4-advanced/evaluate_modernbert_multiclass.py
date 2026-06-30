#!/usr/bin/env python3
"""
Evaluate trained multi-class ModernBERT model on external datasets.

This script loads the fine-tuned 4-class ModernBERT model and tests it on:
- GameTox (baseline test set)
- Civil Comments (Wikipedia/news)
- Real Toxicity Prompts (web text)
- Surge AI (social media)

For binary comparison, classes 1-3 (Profanity, Insult, Hate Speech) are treated as toxic.

Goal: Compare multi-class vs binary ModernBERT performance.
"""

import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 70)
print("MODERNBERT MULTI-CLASS EXTERNAL DATASET EVALUATION")
print("=" * 70)

# ============================================================================
# Load Trained Multi-Class ModernBERT Model
# ============================================================================

print("\nLoading trained multi-class ModernBERT model...")
model_path = "./modernbert_multiclass/run1_4class/final_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Use MPS if available (Apple Silicon)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    print(f"✓ Loaded multi-class ModernBERT from {model_path}")
    print(f"✓ Device: {device}")
    print(f"✓ Number of labels: {model.config.num_labels}")
except Exception as e:
    print(f"✗ Error loading ModernBERT: {e}")
    exit(1)

# ============================================================================
# Prediction Function
# ============================================================================

def predict_modernbert_multiclass(texts, batch_size=32):
    """Make predictions with fine-tuned multi-class ModernBERT.

    Returns:
        predictions_4class: 4-class predictions (0=Clean, 1=Profanity, 2=Insult, 3=Hate Speech)
        predictions_binary: Binary predictions (0=Clean, 1=Toxic)
    """
    predictions_4class = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Multi-class: argmax to get class (0, 1, 2, or 3)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions_4class.extend(batch_preds)

    predictions_4class = np.array(predictions_4class)

    # Convert to binary: 0=Clean, 1-3=Toxic
    predictions_binary = (predictions_4class > 0).astype(int)

    return predictions_4class, predictions_binary

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_on_dataset(dataset_name, df):
    """Evaluate multi-class ModernBERT on a single dataset."""

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    try:
        # Get texts and labels
        texts = df['text'].tolist()
        y_true_binary = df['is_toxic'].values

        print(f"  Samples: {len(texts)}")
        toxic_pct = (y_true_binary.sum() / len(y_true_binary)) * 100
        print(f"  Toxic: {y_true_binary.sum()} ({toxic_pct:.1f}%)")

        # Predict
        print(f"  Running predictions...")
        start_time = time.time()
        y_pred_4class, y_pred_binary = predict_modernbert_multiclass(texts)
        elapsed = time.time() - start_time

        # Calculate binary metrics (for comparison with other approaches)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Get confusion matrix for binary
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Calculate latency per message
        latency_ms = (elapsed / len(texts)) * 1000

        print(f"\n  Binary Classification Results:")
        print(f"    F1:        {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    Latency:   {latency_ms:.2f}ms/msg")
        print(f"    Total:     {elapsed:.1f}s")
        print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

        # Show multi-class breakdown (informational)
        print(f"\n  Multi-class Breakdown:")
        class_names = ['Clean', 'Profanity', 'Insult', 'Hate Speech']
        for i, name in enumerate(class_names):
            count = (y_pred_4class == i).sum()
            pct = count / len(y_pred_4class) * 100
            print(f"    {name:15s}: {count:5d} ({pct:5.1f}%)")

        return {
            'dataset': dataset_name,
            'approach': 'ModernBERT-MultiClass',
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
        print(f"  ✗ Error: {e}")
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

# GameTox (baseline test set)
print("\n[1/4] Loading GameTox test set...")
try:
    gametox = pd.read_csv('../data/test_subset_200_stratified.csv')
    gametox = gametox.rename(columns={'message': 'text'})
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
    if surge['is_toxic'].dtype == 'object':
        surge['is_toxic'] = surge['is_toxic'].map({'Toxic': 1, 'Not Toxic': 0, 1: 1, 0: 0})
    datasets['Surge_AI'] = surge
    toxic_pct = (surge['is_toxic'].sum() / len(surge)) * 100
    print(f"✓ Loaded {len(surge)} samples ({toxic_pct:.1f}% toxic)")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# Run Evaluations
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING EVALUATIONS")
print("=" * 70)

results = []

for dataset_name, df in datasets.items():
    result = evaluate_on_dataset(dataset_name, df)
    if result:
        results.append(result)

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

# Load existing results and append multi-class ModernBERT results
try:
    existing_results = pd.read_csv('external_dataset_results.csv')
    # Remove old ModernBERT-MultiClass results if they exist
    existing_results = existing_results[existing_results['approach'] != 'ModernBERT-MultiClass']
    # Append new results
    combined_results = pd.concat([existing_results, results_df], ignore_index=True)
    combined_results.to_csv('external_dataset_results.csv', index=False)
    print(f"✓ Updated external_dataset_results.csv with multi-class ModernBERT results")
except Exception as e:
    # If file doesn't exist or error, just save multi-class results
    results_df.to_csv('modernbert_multiclass_results.csv', index=False)
    print(f"✓ Saved multi-class ModernBERT results to modernbert_multiclass_results.csv")
    print(f"  (Note: Could not update external_dataset_results.csv: {e})")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\nMulti-Class ModernBERT Performance (converted to binary):")
for _, row in results_df.iterrows():
    print(f"  {row['dataset']:20s} F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}")

# Calculate average on external datasets (exclude GameTox)
external_results = results_df[results_df['dataset'] != 'GameTox']
if len(external_results) > 0:
    avg_external_f1 = external_results['f1'].mean()
    print(f"\n  {'Average (external)':20s} F1={avg_external_f1:.4f}")

    # Check success criteria
    print("\n" + "-" * 70)
    print("Success Criteria Check:")
    print("-" * 70)

    gametox_f1 = results_df[results_df['dataset'] == 'GameTox']['f1'].values
    if len(gametox_f1) > 0:
        gametox_f1 = gametox_f1[0]
        print(f"  GameTox F1:     {gametox_f1:.4f} (target: > 0.70)")
        if gametox_f1 > 0.70:
            print(f"    ✓ PASS")
        else:
            print(f"    ✗ FAIL")

    print(f"  External F1:    {avg_external_f1:.4f} (target: > 0.68)")
    if avg_external_f1 > 0.68:
        print(f"    ✓ PASS - Multi-class ModernBERT generalizes well!")
    else:
        print(f"    ✗ FAIL - Multi-class ModernBERT does not generalize as well as expected")

# Compare all approaches
try:
    all_results = pd.read_csv('external_dataset_results.csv')
    print("\n" + "-" * 70)
    print("Comparison: Binary vs Multi-Class ModernBERT:")
    print("-" * 70)

    for approach in ['ModernBERT', 'ModernBERT-MultiClass']:
        approach_results = all_results[all_results['approach'] == approach]
        if len(approach_results) > 0:
            gametox = approach_results[approach_results['dataset'] == 'GameTox']['f1'].values
            external = approach_results[approach_results['dataset'] != 'GameTox']['f1'].mean()

            if len(gametox) > 0:
                print(f"\n  {approach}:")
                print(f"    GameTox:  F1={gametox[0]:.4f}")
                print(f"    External: F1={external:.4f}")
except Exception as e:
    print(f"\n  (Could not load comparison data: {e})")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
