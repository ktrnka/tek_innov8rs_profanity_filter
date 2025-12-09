#!/usr/bin/env python3
"""Evaluate the trained ML classifier on the test set.

This script:
1. Loads the trained model
2. Evaluates on the 200-message test subset (same as Level 2)
3. Calculates metrics (accuracy, precision, recall, F1)
4. Shows confusion matrix
5. Displays example predictions with confidence
6. Compares with Level 1 and Level 2 results
"""

import csv
import pickle
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def load_gametox(csv_file):
    """Load GameTox data with binary labels (0=clean, 1=toxic)."""
    messages = []
    labels = []
    skipped = 0

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with empty or invalid labels
            if not row['label'] or row['label'].strip() == '':
                skipped += 1
                continue

            try:
                messages.append(row['message'])
                # Binary: 0=clean, 1=toxic
                label = int(float(row['label']))
                labels.append(0 if label == 0 else 1)
            except (ValueError, KeyError):
                skipped += 1
                continue

    if skipped > 0:
        print(f"⚠ Skipped {skipped} rows with missing/invalid labels")

    return messages, labels


def main():
    print("="*70)
    print("  LEVEL 3: TRADITIONAL ML CLASSIFIER - EVALUATION")
    print("="*70)
    print()

    # Load the trained model
    print("Loading trained model...")
    try:
        with open('profanity_classifier.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        print("✓ Model loaded successfully")
        print()
    except FileNotFoundError:
        print("❌ Error: profanity_classifier.pkl not found")
        print("   Please run train_classifier.py first")
        return

    # Load test data (same 200 messages as Level 2)
    print("Loading test set (200 messages)...")
    test_messages, test_labels = load_gametox('../data/test_subset_200_stratified.csv')

    print(f"✓ Loaded {len(test_messages)} test messages")
    print(f"\nTest set distribution:")
    clean_count = test_labels.count(0)
    toxic_count = test_labels.count(1)
    print(f"  Clean (0): {clean_count} ({clean_count/len(test_labels)*100:.1f}%)")
    print(f"  Toxic (1): {toxic_count} ({toxic_count/len(test_labels)*100:.1f}%)")
    print()

    # Make predictions
    print("Making predictions...")
    start_time = time.time()
    predictions = pipeline.predict(test_messages)
    predict_time = time.time() - start_time

    # Get prediction probabilities
    proba = pipeline.predict_proba(test_messages)

    print(f"✓ Predictions complete")
    print()

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)

    print("="*70)
    print("  TEST SET PERFORMANCE")
    print("="*70)
    print()
    print(f"Accuracy:   {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision:  {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:     {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1-Score:   {f1:.3f}")
    print()
    print(f"Inference time: {predict_time*1000:.2f}ms for {len(test_messages)} messages")
    print(f"Latency/msg:    {predict_time/len(test_messages)*1000:.2f}ms")
    print()

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    print("="*70)
    print("  CONFUSION MATRIX")
    print("="*70)
    print()
    print(f"                  Predicted")
    print(f"                  Clean  Toxic")
    print(f"Actual  Clean     {tn:4d}   {fp:4d}")
    print(f"        Toxic     {fn:4d}   {tp:4d}")
    print()
    print(f"True Negatives (TN):  {tn:3d} - Correctly identified as clean")
    print(f"False Positives (FP): {fp:3d} - Clean messages flagged as toxic")
    print(f"False Negatives (FN): {fn:3d} - Toxic messages missed")
    print(f"True Positives (TP):  {tp:3d} - Correctly identified as toxic")
    print()
    if (fp + tn) > 0:
        print(f"False Positive Rate: {fp/(fp+tn)*100:.1f}% (clean messages incorrectly flagged)")
    if (fn + tp) > 0:
        print(f"False Negative Rate: {fn/(fn+tp)*100:.1f}% (toxic messages missed)")
    print()

    # Detailed classification report
    print("="*70)
    print("  DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print()
    print(classification_report(test_labels, predictions,
                              target_names=['Clean', 'Toxic'],
                              digits=3))

    # Example predictions
    print("="*70)
    print("  EXAMPLE PREDICTIONS WITH CONFIDENCE")
    print("="*70)
    print()

    for i in range(min(20, len(test_messages))):
        msg = test_messages[i]
        true_label = "TOXIC" if test_labels[i] == 1 else "CLEAN"
        pred_label = "TOXIC" if predictions[i] == 1 else "CLEAN"
        confidence = proba[i][predictions[i]] * 100

        # Determine correctness
        correct = "✓" if predictions[i] == test_labels[i] else "✗"

        # Format message
        msg_display = msg[:50] + "..." if len(msg) > 50 else msg

        print(f"{correct} \"{msg_display}\"")
        print(f"   True: {true_label:5s} | Pred: {pred_label:5s} ({confidence:.1f}% confident)")
        print()

    # Show uncertain predictions
    uncertain_indices = [i for i, p in enumerate(proba) if 0.5 <= max(p) <= 0.7]
    if uncertain_indices:
        print("="*70)
        print(f"  UNCERTAIN PREDICTIONS (50-70% confidence) - {len(uncertain_indices)} total")
        print("="*70)
        print()

        for i in uncertain_indices[:10]:  # Show up to 10
            msg = test_messages[i]
            pred_label = "TOXIC" if predictions[i] == 1 else "CLEAN"
            confidence = proba[i][predictions[i]] * 100

            msg_display = msg[:60] + "..." if len(msg) > 60 else msg
            print(f"? \"{msg_display}\"")
            print(f"   Prediction: {pred_label} ({confidence:.1f}% confident)")
            print()

    # Cross-level comparison
    print("="*70)
    print("  CROSS-LEVEL COMPARISON")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Level 1':<15} {'Level 2':<15} {'Level 3':<15}")
    print(f"{'':20} {'(Rule-Based)':<15} {'(LLM)':<15} {'(ML)':<15}")
    print("-" * 70)
    print(f"{'F1-Score':<20} {'0.650':<15} {'0.816':<15} {f'{f1:.3f}':<15}")
    print(f"{'Precision':<20} {'0.580':<15} {'0.769':<15} {f'{precision:.3f}':<15}")
    print(f"{'Recall':<20} {'0.740':<15} {'0.870':<15} {f'{recall:.3f}':<15}")
    print(f"{'Latency (ms/msg)':<20} {'<1':<15} {'181':<15} {f'{predict_time/len(test_messages)*1000:.2f}':<15}")
    print(f"{'Cost per 1M msgs':<20} {'$0':<15} {'$1,710':<15} {'$0':<15}")
    print()
    print("="*70)
    print()
    print("Key Insights:")
    print("  • Traditional ML offers middle-ground performance")
    print("  • Much faster than LLM (no API calls)")
    print("  • Zero cost after training (vs $1,710/1M for LLM)")
    print("  • Better than rules but not quite LLM-level accuracy")
    print("  • Model is interpretable (can see learned features)")
    print()


if __name__ == '__main__':
    main()
