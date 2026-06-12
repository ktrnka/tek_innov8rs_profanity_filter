#!/usr/bin/env python3
"""Test Level 1 regex detector on the 200-message stratified sample for fair comparison with Level 2."""

import csv
import re
import sys
sys.path.insert(0, '../level1-rule-based')

from regex_detector_expanded import RegexProfanityDetector

# Expanded word list from Level 1 (best performing configuration)
EXPANDED_WORDS = [
    # Original 7 words
    'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell',
    # Expanded words from analysis
    'idiot', 'idiots',
    'wtf',
    'fucking',
    'useless',
    'stupid',
    'retard', 'retards',
    'moron', 'morons',
    'ffs',
    'fck',
    'stfu',
    'trash',
    'dumb',
    'noob', 'noobs',
    'bot', 'bots',
    'camper', 'campers',
    'camping'
]

def load_messages(csv_file):
    """Load messages and labels from CSV."""
    messages = []
    labels = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row['message'])
            labels.append(float(row['label']))

    return messages, labels

def calculate_metrics(y_true, y_pred):
    """Calculate TP, FP, TN, FN and metrics."""
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 0)

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flagged': tp + fp
    }

def main():
    print("="*70)
    print("  LEVEL 1 BASELINE: Testing on 200-message stratified sample")
    print("="*70)
    print()

    # Load the 200-message sample
    messages, labels = load_messages('../data/test_subset_200_stratified.csv')

    print(f"Loaded {len(messages)} messages")
    toxic_count = sum(1 for l in labels if l == 1.0)
    clean_count = len(labels) - toxic_count
    print(f"  Toxic: {toxic_count} ({toxic_count/len(labels)*100:.1f}%)")
    print(f"  Clean: {clean_count} ({clean_count/len(labels)*100:.1f}%)")
    print()

    # Test with normalization (best Level 1 config)
    print("Testing Level 1 with normalization...")
    detector = RegexProfanityDetector(EXPANDED_WORDS, use_normalization=True)

    predictions = [1 if detector.is_profane(msg) else 0 for msg in messages]
    metrics = calculate_metrics(labels, predictions)

    print()
    print("="*70)
    print("  LEVEL 1 RESULTS (200-message sample)")
    print("="*70)
    print(f"Messages evaluated: {len(messages)}")
    print(f"Messages flagged: {metrics['flagged']} ({metrics['flagged']/len(messages)*100:.1f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']:>3}  |  FP: {metrics['fp']:>3}")
    print(f"  FN: {metrics['fn']:>3}  |  TN: {metrics['tn']:>3}")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1-score:  {metrics['f1']:.3f}")
    print()
    print("✓ Level 1 baseline established on 200-message sample")
    print()

    # Save results for comparison
    import pickle
    with open('level1_baseline_results.pkl', 'wb') as f:
        pickle.dump({
            'metrics': metrics,
            'predictions': predictions,
            'labels': labels,
            'messages': messages
        }, f)
    print("💾 Results saved to level1_baseline_results.pkl")

if __name__ == '__main__':
    main()
