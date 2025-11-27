#!/usr/bin/env python3
"""
Level 1 - Task 3: Regex-based profanity detector
Uses word boundaries and a small list of profane words.
"""

import csv
import re
from pathlib import Path

def load_gametox(data_path):
    """Load GameTox dataset."""
    messages = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing labels
            if not row['label'] or row['label'].strip() == '':
                continue
            messages.append(row['message'])
            labels.append(float(row['label']))

    return messages, labels

class RegexProfanityDetector:
    """Simple regex-based profanity detector."""

    def __init__(self, profane_words):
        """
        Initialize detector with list of profane words.

        Args:
            profane_words: List of words to detect
        """
        self.profane_words = profane_words
        # Create regex pattern with word boundaries to avoid substring matches
        # \b ensures we match whole words
        pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def is_profane(self, text):
        """Check if text contains profanity."""
        return bool(self.pattern.search(text))

    def detect_batch(self, messages):
        """Detect profanity in batch of messages."""
        return [self.is_profane(msg) for msg in messages]

def evaluate_detector(detector, messages, labels):
    """Evaluate detector performance."""
    predictions = detector.detect_batch(messages)

    total = len(messages)
    flagged = sum(predictions)

    # Binary classification: 0 = clean, >0 = toxic
    actual_toxic = [label > 0 for label in labels]

    # Calculate confusion matrix values
    true_positives = sum(1 for i in range(total) if predictions[i] and actual_toxic[i])
    false_positives = sum(1 for i in range(total) if predictions[i] and not actual_toxic[i])
    true_negatives = sum(1 for i in range(total) if not predictions[i] and not actual_toxic[i])
    false_negatives = sum(1 for i in range(total) if not predictions[i] and actual_toxic[i])

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'total': total,
        'flagged': flagged,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def print_results(results, word_list):
    """Print evaluation results."""
    print(f"=== Profanity Detector Results ===")
    print(f"Word list ({len(word_list)} words): {', '.join(word_list)}")
    print()
    print(f"Total messages: {results['total']:,}")
    print(f"Messages flagged: {results['flagged']:,} ({100 * results['flagged'] / results['total']:.2f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {results['true_positives']:,} (correctly flagged as toxic)")
    print(f"  False Positives: {results['false_positives']:,} (incorrectly flagged as toxic)")
    print(f"  True Negatives:  {results['true_negatives']:,} (correctly identified as clean)")
    print(f"  False Negatives: {results['false_negatives']:,} (missed toxic messages)")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
    print(f"  Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
    print(f"  F1-score:  {results['f1']:.3f}")
    print()
    print("Interpretation:")
    print(f"  - Of flagged messages, {results['precision']*100:.1f}% are actually toxic (precision)")
    print(f"  - Of toxic messages, we catch {results['recall']*100:.1f}% of them (recall)")

def show_examples(detector, messages, labels, n=10):
    """Show example flagged messages."""
    print("\n=== Sample Flagged Messages ===")
    predictions = detector.detect_batch(messages)
    count = 0
    for i, is_flagged in enumerate(predictions):
        if is_flagged:
            toxic_label = "TOXIC" if labels[i] > 0 else "CLEAN"
            print(f"[{toxic_label}] {messages[i]}")
            count += 1
            if count >= n:
                break

def main():
    # Path to dataset
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    # Start with 7 common profane words
    profane_words = [
        'damn',
        'shit',
        'fuck',
        'ass',
        'bitch',
        'bastard',
        'hell'
    ]

    print("Loading GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"Loaded {len(messages):,} messages\n")

    # Create and evaluate detector
    detector = RegexProfanityDetector(profane_words)
    results = evaluate_detector(detector, messages, labels)

    # Print results
    print_results(results, profane_words)

    # Show examples
    show_examples(detector, messages, labels)

if __name__ == "__main__":
    main()
