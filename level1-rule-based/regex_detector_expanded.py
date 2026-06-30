#!/usr/bin/env python3
"""
Level 1 - Expanded regex-based profanity detector with text normalization
Uses expanded word list based on analysis of false negatives.
Now includes text normalization to handle bypass attempts (leetspeak, spacing, etc.)
"""

import csv
import re
from pathlib import Path
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

        # Check partial masking first (before normalization)
        if self.normalizer and self.normalizer.detect_partial_masking(text, self.profane_words):
            return True

        # If normalization is disabled, just do simple regex check
        if not self.use_normalization:
            return bool(self.pattern.search(text))

        # Normalize text while preserving word boundaries
        # This allows regex with \b to work correctly
        normalized = self.normalizer.normalize_preserving_boundaries(text)
        return bool(self.pattern.search(normalized))

    def detect_batch(self, messages):
        return [self.is_profane(msg) for msg in messages]

def evaluate_detector(detector, messages, labels):
    """Evaluate detector performance."""
    predictions = detector.detect_batch(messages)
    total = len(messages)
    actual_toxic = [label > 0 for label in labels]

    true_positives = sum(1 for i in range(total) if predictions[i] and actual_toxic[i])
    false_positives = sum(1 for i in range(total) if predictions[i] and not actual_toxic[i])
    true_negatives = sum(1 for i in range(total) if not predictions[i] and not actual_toxic[i])
    false_negatives = sum(1 for i in range(total) if not predictions[i] and actual_toxic[i])

    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'flagged': sum(predictions),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_results(results, word_list, title="Results"):
    """Print evaluation results."""
    print(f"\n=== {title} ===")
    print(f"Word list size: {len(word_list)} words")
    print(f"Total messages: {results['total']:,}")
    print(f"Messages flagged: {results['flagged']:,} ({100 * results['flagged'] / results['total']:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {results['true_positives']:,}  |  FP: {results['false_positives']:,}")
    print(f"  FN: {results['false_negatives']:,}  |  TN: {results['true_negatives']:,}")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
    print(f"  Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
    print(f"  F1-score:  {results['f1']:.3f}")

def main():
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    # Original word list (7 words)
    original_words = [
        'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell'
    ]

    # Expanded word list based on false negative analysis
    # Adding top profanity/insults from missed messages
    expanded_words = original_words + [
        # Top insults from analysis
        'idiot', 'idiots',
        'wtf',
        'fucking',
        'useless',
        'stupid',
        'retard', 'retards',
        'moron', 'morons',
        'ffs',
        'fck',  # variant of fuck
        'stfu',
        'trash',
        'dumb',
        # Additional common gaming insults
        'noob', 'noobs',
        'bot', 'bots',
        'camper', 'campers',
        'camping'
    ]

    print("Loading GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"Loaded {len(messages):,} messages")

    # Test original detector
    print("\n" + "="*60)
    original_detector = RegexProfanityDetector(original_words)
    original_results = evaluate_detector(original_detector, messages, labels)
    print_results(original_results, original_words, "Original Detector (7 words)")

    # Test expanded detector
    print("\n" + "="*60)
    expanded_detector = RegexProfanityDetector(expanded_words)
    expanded_results = evaluate_detector(expanded_detector, messages, labels)
    print_results(expanded_results, expanded_words, "Expanded Detector (28 words)")

    # Show improvement
    print("\n" + "="*60)
    print("=== Impact of Expansion ===")
    print(f"Words added: {len(expanded_words) - len(original_words)} ({len(original_words)} → {len(expanded_words)})")
    print()
    print("Changes in metrics:")
    print(f"  Recall:    {original_results['recall']:.3f} → {expanded_results['recall']:.3f} (Δ {expanded_results['recall'] - original_results['recall']:+.3f})")
    print(f"  Precision: {original_results['precision']:.3f} → {expanded_results['precision']:.3f} (Δ {expanded_results['precision'] - original_results['precision']:+.3f})")
    print(f"  F1-score:  {original_results['f1']:.3f} → {expanded_results['f1']:.3f} (Δ {expanded_results['f1'] - original_results['f1']:+.3f})")
    print(f"  Accuracy:  {original_results['accuracy']:.3f} → {expanded_results['accuracy']:.3f} (Δ {expanded_results['accuracy'] - original_results['accuracy']:+.3f})")
    print()
    print("Toxic messages caught:")
    print(f"  Original:  {original_results['true_positives']:,} / {original_results['true_positives'] + original_results['false_negatives']:,} ({100 * original_results['recall']:.1f}%)")
    print(f"  Expanded:  {expanded_results['true_positives']:,} / {expanded_results['true_positives'] + expanded_results['false_negatives']:,} ({100 * expanded_results['recall']:.1f}%)")
    print(f"  Additional toxic messages caught: {expanded_results['true_positives'] - original_results['true_positives']:,}")
    print()
    print("⚠️  PRECISION/RECALL TRADEOFF:")
    if expanded_results['recall'] > original_results['recall'] and expanded_results['precision'] < original_results['precision']:
        print("  ✓ Recall improved (catching more toxic messages)")
        print("  ✗ Precision decreased (more false positives)")
        print("  → This is the classic tradeoff!")
    elif expanded_results['recall'] > original_results['recall'] and expanded_results['precision'] >= original_results['precision']:
        print("  ✓ Both recall AND precision improved!")
        print("  → Our word selection was effective")

    # Show word list
    print("\n" + "="*60)
    print("=== Expanded Word List ===")
    print(", ".join(expanded_words))

    # Test normalization impact
    print("\n" + "="*60)
    print("="*60)
    print("  TEXT NORMALIZATION ENHANCEMENT")
    print("="*60)
    print("Testing impact of text normalization on bypass detection")
    print()

    # Test with normalization enabled
    normalized_detector = RegexProfanityDetector(expanded_words, use_normalization=True)
    normalized_results = evaluate_detector(normalized_detector, messages, labels)
    print_results(normalized_results, expanded_words, "With Text Normalization")

    # Compare: Expanded (no norm) vs Normalized
    print("\n" + "="*60)
    print("=== Impact of Text Normalization ===")
    print(f"\n{'Metric':<12} {'Without Norm':<15} {'With Norm':<15} {'Change':<20}")
    print("-"*60)

    metrics = [
        ('Recall', 'recall'),
        ('Precision', 'precision'),
        ('F1-score', 'f1'),
        ('Accuracy', 'accuracy')
    ]

    for label, key in metrics:
        without = expanded_results[key]
        with_norm = normalized_results[key]
        delta = with_norm - without
        delta_pct = (delta / without * 100) if without > 0 else 0
        symbol = "📈" if delta > 0.001 else "📉" if delta < -0.001 else "➡️"

        print(f"{label:<12} {without:.3f} ({without*100:5.1f}%)  {with_norm:.3f} ({with_norm*100:5.1f}%)  {symbol} {delta:+.3f} ({delta_pct:+5.1f}%)")

    print()
    print("Detection changes:")
    print(f"  Flagged without normalization: {expanded_results['flagged']:,}")
    print(f"  Flagged with normalization:    {normalized_results['flagged']:,}")
    print(f"  Additional messages flagged:   {normalized_results['flagged'] - expanded_results['flagged']:,}")

    print()
    print("False positives:")
    print(f"  Without normalization: {expanded_results['false_positives']:,}")
    print(f"  With normalization:    {normalized_results['false_positives']:,}")
    fp_change = normalized_results['false_positives'] - expanded_results['false_positives']
    print(f"  Change: {fp_change:+,} ({'+' if fp_change >= 0 else ''}{100*fp_change/expanded_results['false_positives'] if expanded_results['false_positives'] > 0 else 0:.1f}%)")

    # Demonstrate on bypass examples
    print("\n" + "="*60)
    print("=== Bypass Detection Examples ===")
    print()

    bypass_examples = [
        "You're such an 1d10t",  # leetspeak
        "sh1t happens",
        "what the f u c k",  # spacing
        "Go to h3ll",
        "f*** you",  # partial masking
        "You аss",  # Cyrillic homoglyph
        "That's shiiiit",  # repeated chars
        "d.a.m.n it",  # character insertion
    ]

    print(f"{'Message':<30} {'Without Norm':<15} {'With Norm':<15}")
    print("-"*60)
    for example in bypass_examples:
        without_detect = expanded_detector.is_profane(example)
        with_detect = normalized_detector.is_profane(example)
        without_str = "🚫 BLOCKED" if without_detect else "✅ MISSED"
        with_str = "🚫 BLOCKED" if with_detect else "✅ MISSED"
        print(f"{example:<30} {without_str:<15} {with_str:<15}")

    print()
    print("💡 Key Insight: Text normalization catches bypass attempts that")
    print("   simple regex misses (leetspeak, spacing, Unicode, etc.)")

if __name__ == "__main__":
    main()
