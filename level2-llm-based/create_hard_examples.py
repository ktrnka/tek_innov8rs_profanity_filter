#!/usr/bin/env python3
"""
Extract hard examples from full GameTox dataset using Level 1 predictions.
Creates test_subset_hard_100.csv with 100 challenging examples for Level 2 testing.
"""

import csv
import sys
from pathlib import Path

# Add level1 to path to import detector
sys.path.insert(0, str(Path(__file__).parent.parent / "level1-rule-based"))
from regex_detector_expanded import RegexProfanityDetector, load_gametox


def extract_hard_examples(messages, labels, predictions, num_examples=100):
    """
    Extract hard examples (false positives and false negatives).

    Returns:
        List of (message, label, prediction, error_type) tuples
    """
    hard_examples = []

    for i, (msg, label, pred) in enumerate(zip(messages, labels, predictions)):
        actual_toxic = label > 0

        # False positive: flagged as toxic but actually clean
        if pred and not actual_toxic:
            hard_examples.append({
                'index': i,
                'message': msg,
                'label': label,
                'level1_prediction': 1,
                'error_type': 'false_positive'
            })

        # False negative: missed actual toxic content
        elif not pred and actual_toxic:
            hard_examples.append({
                'index': i,
                'message': msg,
                'label': label,
                'level1_prediction': 0,
                'error_type': 'false_negative'
            })

    return hard_examples


def select_balanced_sample(hard_examples, num_examples=100):
    """
    Select balanced sample of false positives and false negatives.
    """
    fps = [ex for ex in hard_examples if ex['error_type'] == 'false_positive']
    fns = [ex for ex in hard_examples if ex['error_type'] == 'false_negative']

    print(f"\nHard examples found:")
    print(f"  False positives: {len(fps):,} (clean messages wrongly flagged)")
    print(f"  False negatives: {len(fns):,} (toxic messages missed)")

    # Balance the sample: 50 FP + 50 FN
    # If we don't have enough of one type, take more from the other
    num_fps = min(len(fps), num_examples // 2)
    num_fns = min(len(fns), num_examples // 2)

    # If one category is short, make up with the other
    if num_fps < num_examples // 2:
        num_fns = min(len(fns), num_examples - num_fps)
    elif num_fns < num_examples // 2:
        num_fps = min(len(fps), num_examples - num_fns)

    selected = fps[:num_fps] + fns[:num_fns]

    print(f"\nSelected {len(selected)} examples:")
    print(f"  False positives: {num_fps}")
    print(f"  False negatives: {num_fns}")

    return selected


def save_hard_examples(examples, output_path):
    """Save hard examples to CSV."""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'message', 'label', 'level1_prediction', 'error_type'])
        writer.writeheader()
        writer.writerows(examples)

    print(f"\n✓ Saved to {output_path}")


def main():
    print("="*70)
    print("  EXTRACTING HARD EXAMPLES FROM LEVEL 1")
    print("="*70)

    # Load full GameTox dataset
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    if not data_path.exists():
        print(f"\n❌ ERROR: GameTox dataset not found at {data_path}")
        return

    print(f"\nLoading full GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"✓ Loaded {len(messages):,} messages")

    # Initialize Level 1 detector (best version: expanded + normalization)
    print(f"\nInitializing Level 1 detector (expanded words + normalization)...")

    expanded_words = [
        # Original 7 words
        'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell',
        # Expanded words
        'idiot', 'idiots', 'wtf', 'fucking', 'useless', 'stupid',
        'retard', 'retards', 'moron', 'morons', 'ffs', 'fck',
        'stfu', 'trash', 'dumb', 'noob', 'noobs', 'bot', 'bots',
        'camper', 'campers', 'camping'
    ]

    detector = RegexProfanityDetector(expanded_words, use_normalization=True)
    print(f"✓ Detector initialized ({len(expanded_words)} words, normalization enabled)")

    # Run predictions on full dataset
    print(f"\nRunning Level 1 on {len(messages):,} messages...")
    predictions = detector.detect_batch(messages)
    print(f"✓ Predictions complete")

    # Calculate overall metrics
    actual_toxic = [label > 0 for label in labels]
    tp = sum(1 for i in range(len(messages)) if predictions[i] and actual_toxic[i])
    fp = sum(1 for i in range(len(messages)) if predictions[i] and not actual_toxic[i])
    tn = sum(1 for i in range(len(messages)) if not predictions[i] and not actual_toxic[i])
    fn = sum(1 for i in range(len(messages)) if not predictions[i] and actual_toxic[i])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nLevel 1 performance on full dataset:")
    print(f"  Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"  F1-score:  {f1:.3f}")
    print(f"\nConfusion matrix:")
    print(f"  TP: {tp:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TN: {tn:,}")

    # Extract hard examples
    print(f"\n{'='*70}")
    print("EXTRACTING HARD EXAMPLES")
    print(f"{'='*70}")

    hard_examples = extract_hard_examples(messages, labels, predictions, num_examples=100)
    selected = select_balanced_sample(hard_examples, num_examples=100)

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "test_subset_hard_100.csv"
    save_hard_examples(selected, output_path)

    print("\n✓ Hard examples extraction complete!")
    print(f"\nNext steps:")
    print(f"  1. Review {output_path.name}")
    print(f"  2. Test Level 2 (LLM) on these 100 hard examples")
    print(f"  3. Compare if LLMs can solve cases where regex fails")
    print(f"\nThis directly tests: 'Can LLMs handle what regex can't?'")


if __name__ == "__main__":
    main()
