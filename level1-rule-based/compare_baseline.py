#!/usr/bin/env python3
"""
Compare our regex detector against alt-profanity-check baseline.
Tests both non-normalized and normalized versions to show progression.
"""

import csv
import re
from pathlib import Path
from profanity_check import predict
from text_normalizer import TextNormalizer

def load_gametox(data_path):
    """Load GameTox dataset."""
    messages = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['label'] or row['label'].strip() == '':
                continue
            messages.append(row['message'])
            labels.append(float(row['label']))

    return messages, labels

class RegexProfanityDetector:
    """Regex-based profanity detector with optional text normalization."""

    def __init__(self, profane_words, use_normalization=False):
        self.profane_words = profane_words
        self.use_normalization = use_normalization
        self.normalizer = TextNormalizer() if use_normalization else None
        pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def is_profane(self, text):
        if not text:
            return False

        # Check partial masking first
        if self.normalizer and self.normalizer.detect_partial_masking(text, self.profane_words):
            return True

        if not self.use_normalization:
            return bool(self.pattern.search(text))

        # Use boundary-preserving normalization
        normalized = self.normalizer.normalize_preserving_boundaries(text)
        return bool(self.pattern.search(normalized))

    def predict_batch(self, messages):
        """Return 1 for profane, 0 for clean."""
        return [1 if self.is_profane(msg) else 0 for msg in messages]

def evaluate(predictions, labels):
    """Calculate metrics."""
    total = len(labels)
    actual_toxic = [1 if label > 0 else 0 for label in labels]

    tp = sum(1 for i in range(total) if predictions[i] == 1 and actual_toxic[i] == 1)
    fp = sum(1 for i in range(total) if predictions[i] == 1 and actual_toxic[i] == 0)
    tn = sum(1 for i in range(total) if predictions[i] == 0 and actual_toxic[i] == 0)
    fn = sum(1 for i in range(total) if predictions[i] == 0 and actual_toxic[i] == 1)

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flagged': tp + fp
    }

def print_comparison(our_results, baseline_results, total_messages):
    """Print comparison table."""
    print("\n" + "="*80)
    print("=== COMPARISON: Our Detector vs. alt-profanity-check ===")
    print("="*80)

    print(f"\nDataset: {total_messages:,} messages\n")

    print(f"{'Metric':<20} {'Our Detector':>15} {'alt-profanity':>15} {'Difference':>15}")
    print("-" * 80)

    metrics = [
        ('Messages Flagged', our_results['flagged'], baseline_results['flagged']),
        ('Accuracy', our_results['accuracy'], baseline_results['accuracy']),
        ('Precision', our_results['precision'], baseline_results['precision']),
        ('Recall', our_results['recall'], baseline_results['recall']),
        ('F1-Score', our_results['f1'], baseline_results['f1']),
    ]

    for metric_name, ours, baseline in metrics:
        if metric_name == 'Messages Flagged':
            diff = ours - baseline
            print(f"{metric_name:<20} {ours:>15,} {baseline:>15,} {diff:>+15,}")
        else:
            diff = ours - baseline
            print(f"{metric_name:<20} {ours:>15.3f} {baseline:>15.3f} {diff:>+15.3f}")

    print("\n" + "="*80)
    print("=== Confusion Matrix Comparison ===")
    print("="*80)

    print(f"\n{'':20} {'Our Detector':^30} {'alt-profanity-check':^30}")
    print(f"{'':20} {'TP':>10} {'FP':>10} {'TN':>10} {'FN':>10}   {'TP':>10} {'FP':>10} {'TN':>10} {'FN':>10}")
    print("-" * 80)
    print(f"{'Values':<20} {our_results['tp']:>10,} {our_results['fp']:>10,} {our_results['tn']:>10,} {our_results['fn']:>10,}   {baseline_results['tp']:>10,} {baseline_results['fp']:>10,} {baseline_results['tn']:>10,} {baseline_results['fn']:>10,}")

    print("\n" + "="*80)
    print("=== Analysis ===")
    print("="*80)

    if our_results['f1'] > baseline_results['f1']:
        print("\n✓ Our detector has BETTER F1-score than the baseline!")
        print(f"  F1: {our_results['f1']:.3f} vs {baseline_results['f1']:.3f} (+{our_results['f1'] - baseline_results['f1']:.3f})")
    else:
        print("\n✗ Baseline has better F1-score")
        print(f"  F1: {baseline_results['f1']:.3f} vs {our_results['f1']:.3f} (+{baseline_results['f1'] - our_results['f1']:.3f})")

    print("\nTradeoffs:")
    if our_results['precision'] > baseline_results['precision']:
        print(f"  ✓ Our precision is higher ({our_results['precision']:.1%} vs {baseline_results['precision']:.1%})")
        print(f"    → Fewer false positives")
    else:
        print(f"  ✗ Baseline precision is higher ({baseline_results['precision']:.1%} vs {our_results['precision']:.1%})")

    if our_results['recall'] > baseline_results['recall']:
        print(f"  ✓ Our recall is higher ({our_results['recall']:.1%} vs {baseline_results['recall']:.1%})")
        print(f"    → Catch more toxic messages")
    else:
        print(f"  ✗ Baseline recall is higher ({baseline_results['recall']:.1%} vs {our_results['recall']:.1%})")
        print(f"    → Baseline catches more toxic messages")

    print("\nKey Differences:")
    print("  Our detector: Rule-based (29 profane words with word boundaries)")
    print("  alt-profanity-check: ML-based (Linear SVM trained on 200k samples)")

def print_three_way_comparison(no_norm_results, normalized_results, baseline_results, total_messages):
    """Print three-way comparison table."""
    print("\n" + "="*90)
    print("=== THREE-WAY COMPARISON ===")
    print("="*90)
    print("\nShowing progression: Basic → Normalized → ML Baseline")
    print(f"Dataset: {total_messages:,} messages\n")

    print(f"{'Metric':<20} {'No Norm':>15} {'Normalized':>15} {'alt-profanity':>15} {'Best':>15}")
    print("-" * 90)

    metrics = [
        ('Messages Flagged', no_norm_results['flagged'], normalized_results['flagged'], baseline_results['flagged']),
        ('Accuracy', no_norm_results['accuracy'], normalized_results['accuracy'], baseline_results['accuracy']),
        ('Precision', no_norm_results['precision'], normalized_results['precision'], baseline_results['precision']),
        ('Recall', no_norm_results['recall'], normalized_results['recall'], baseline_results['recall']),
        ('F1-Score', no_norm_results['f1'], normalized_results['f1'], baseline_results['f1']),
    ]

    for metric_name, no_norm, normalized, baseline in metrics:
        if metric_name == 'Messages Flagged':
            best = "—"
            print(f"{metric_name:<20} {no_norm:>15,} {normalized:>15,} {baseline:>15,} {best:>15}")
        else:
            # Determine best value
            best_val = max(no_norm, normalized, baseline)
            if best_val == normalized:
                best = "Normalized ✓"
            elif best_val == no_norm:
                best = "No Norm"
            else:
                best = "alt-profanity"

            print(f"{metric_name:<20} {no_norm:>15.3f} {normalized:>15.3f} {baseline:>15.3f} {best:>15}")

    print("\n" + "="*90)
    print("=== KEY COMPARISONS ===")
    print("="*90)

    print("\n1. Impact of Normalization (Our Detector):")
    print(f"   Precision: {no_norm_results['precision']:.3f} → {normalized_results['precision']:.3f} ({normalized_results['precision'] - no_norm_results['precision']:+.3f})")
    print(f"   Recall:    {no_norm_results['recall']:.3f} → {normalized_results['recall']:.3f} ({normalized_results['recall'] - no_norm_results['recall']:+.3f})")
    print(f"   F1-Score:  {no_norm_results['f1']:.3f} → {normalized_results['f1']:.3f} ({normalized_results['f1'] - no_norm_results['f1']:+.3f})")

    print("\n2. Our Normalized Detector vs. alt-profanity-check:")
    print(f"   Precision: {normalized_results['precision']:.3f} vs {baseline_results['precision']:.3f} ({normalized_results['precision'] - baseline_results['precision']:+.3f})")
    print(f"   Recall:    {normalized_results['recall']:.3f} vs {baseline_results['recall']:.3f} ({normalized_results['recall'] - baseline_results['recall']:+.3f})")
    print(f"   F1-Score:  {normalized_results['f1']:.3f} vs {baseline_results['f1']:.3f} ({normalized_results['f1'] - baseline_results['f1']:+.3f})")

    if normalized_results['f1'] > baseline_results['f1']:
        print("\n   ✓ Our NORMALIZED detector BEATS alt-profanity-check!")
    elif no_norm_results['f1'] > baseline_results['f1']:
        print("\n   ✓ Our detector (even without normalization) BEATS alt-profanity-check!")
    else:
        print("\n   ✗ alt-profanity-check has better F1-score")

    print("\n" + "="*90)
    print("=== SUMMARY ===")
    print("="*90)
    print("\nOur Detector (No Normalization):")
    print("  • Rule-based: 29 profane words with regex word boundaries")
    print(f"  • F1-Score: {no_norm_results['f1']:.3f}")
    print(f"  • Trade-off: High precision ({no_norm_results['precision']:.1%}), moderate recall ({no_norm_results['recall']:.1%})")

    print("\nOur Detector (WITH Normalization):")
    print("  • Same 29 words + text normalization (leetspeak, homoglyphs, etc.)")
    print(f"  • F1-Score: {normalized_results['f1']:.3f}")
    print(f"  • Improvement: {normalized_results['f1'] - no_norm_results['f1']:+.3f} F1 from normalization")

    print("\nalt-profanity-check (ML Baseline):")
    print("  • ML-based: Linear SVM trained on 200k samples")
    print(f"  • F1-Score: {baseline_results['f1']:.3f}")
    print(f"  • General-purpose (not gaming-specific)")

def main():
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    # Our word list
    our_words = [
        'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell',
        'idiot', 'idiots', 'wtf', 'fucking', 'useless', 'stupid',
        'retard', 'retards', 'moron', 'morons', 'ffs', 'fck',
        'stfu', 'trash', 'dumb', 'noob', 'noobs', 'bot', 'bots',
        'camper', 'campers', 'camping'
    ]

    print("="*90)
    print("  COMPREHENSIVE BASELINE COMPARISON")
    print("="*90)
    print("\nTesting three approaches:")
    print("  1. Our detector WITHOUT normalization (Day 1 baseline)")
    print("  2. Our detector WITH normalization (Day 2 enhancement)")
    print("  3. alt-profanity-check (ML baseline)")
    print()

    print("Loading GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"Loaded {len(messages):,} messages\n")

    # Test our detector WITHOUT normalization
    print("1. Evaluating our detector (no normalization)...")
    no_norm_detector = RegexProfanityDetector(our_words, use_normalization=False)
    no_norm_predictions = no_norm_detector.predict_batch(messages)
    no_norm_results = evaluate(no_norm_predictions, labels)

    # Test our detector WITH normalization
    print("2. Evaluating our detector (with normalization)...")
    norm_detector = RegexProfanityDetector(our_words, use_normalization=True)
    norm_predictions = norm_detector.predict_batch(messages)
    norm_results = evaluate(norm_predictions, labels)

    # Test alt-profanity-check
    print("3. Evaluating alt-profanity-check (this may take a minute)...")
    baseline_predictions = predict(messages)
    baseline_results = evaluate(baseline_predictions, labels)

    # Print three-way comparison
    print_three_way_comparison(no_norm_results, norm_results, baseline_results, len(messages))

if __name__ == "__main__":
    main()
