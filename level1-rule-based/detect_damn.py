#!/usr/bin/env python3
"""
Level 1 - Task 2: Single word detection
Detects the word "damn" in GameTox messages and analyzes results.
"""

import csv
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

def detect_word(messages, word):
    """
    Detect if a word appears in messages (case-insensitive).
    Returns list of booleans indicating if word was found.
    """
    word_lower = word.lower()
    return [word_lower in message.lower() for message in messages]

def analyze_results(messages, labels, detections, word):
    """Analyze detection results."""
    total_messages = len(messages)
    flagged_count = sum(detections)

    # Count correct and incorrect flags
    correct_flags = 0  # Flagged AND actually toxic (label > 0)
    incorrect_flags = 0  # Flagged BUT not toxic (label == 0)

    for i, is_flagged in enumerate(detections):
        if is_flagged:
            if labels[i] > 0:  # Toxic
                correct_flags += 1
            else:  # Non-toxic
                incorrect_flags += 1

    # Print results
    print(f"=== Analysis for word: '{word}' ===")
    print(f"Total messages: {total_messages:,}")
    print(f"Messages flagged (containing '{word}'): {flagged_count:,}")
    print(f"Percentage flagged: {100 * flagged_count / total_messages:.2f}%")
    print()
    print(f"Correct flags (toxic messages): {correct_flags:,}")
    print(f"Incorrect flags (non-toxic messages): {incorrect_flags:,}")
    print()

    # Show some examples
    print("=== Sample flagged messages ===")
    count = 0
    for i, is_flagged in enumerate(detections):
        if is_flagged:
            toxic_label = "TOXIC" if labels[i] > 0 else "CLEAN"
            print(f"[{toxic_label}] {messages[i]}")
            count += 1
            if count >= 10:
                break

def main():
    # Path to dataset (relative to project root)
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    print("Loading GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"Loaded {len(messages):,} messages\n")

    # Detect "damn"
    word = "damn"
    detections = detect_word(messages, word)

    # Analyze
    analyze_results(messages, labels, detections, word)

if __name__ == "__main__":
    main()
