#!/usr/bin/env python3
"""
Analyze false negatives to find common words in missed toxic messages.
This helps us build a data-driven word list.
"""

import csv
import re
from pathlib import Path
from collections import Counter

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
    """Simple regex-based profanity detector."""

    def __init__(self, profane_words):
        self.profane_words = profane_words
        pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def is_profane(self, text):
        return bool(self.pattern.search(text))

def tokenize(text):
    """Extract words from text."""
    # Simple word extraction (lowercase, alphanumeric)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words

def analyze_false_negatives(messages, labels, detector):
    """Find and analyze messages we missed (false negatives)."""
    false_negatives = []

    for i, msg in enumerate(messages):
        is_toxic = labels[i] > 0
        was_flagged = detector.is_profane(msg)

        # False negative: toxic but we didn't flag it
        if is_toxic and not was_flagged:
            false_negatives.append(msg)

    return false_negatives

def find_common_words(messages, current_profane_words, top_n=50):
    """Find most common words in messages, excluding current profane words."""
    all_words = []
    for msg in messages:
        all_words.extend(tokenize(msg))

    # Count frequencies
    word_counts = Counter(all_words)

    # Common English stop words that aren't profane
    stop_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
        'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
        'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
        'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
        'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'did',
        'having', 'may', 'am', 'very', 'much', 'too', 'here', 'such', 'where', 'why',
        # Gaming-specific non-profane words
        'team', 'game', 'win', 'lose', 'gg', 'wp', 'glhf', 'ty', 'thx', 'nice', 'good',
        'bad', 'help', 'need', 'go', 'don', 't', 's', 're', 've', 'll', 'noob', 'pro'
    }

    # Filter out stop words and current profane words
    current_lower = {w.lower() for w in current_profane_words}
    filtered_counts = {
        word: count for word, count in word_counts.items()
        if word not in stop_words and word not in current_lower and len(word) > 2
    }

    return Counter(filtered_counts).most_common(top_n)

def main():
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    # Current word list
    current_words = ['damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell']

    print("Loading GameTox dataset...")
    messages, labels = load_gametox(data_path)
    print(f"Loaded {len(messages):,} messages\n")

    # Create detector with current words
    detector = RegexProfanityDetector(current_words)

    # Find false negatives
    print("Finding false negatives (toxic messages we missed)...")
    false_negatives = analyze_false_negatives(messages, labels, detector)
    print(f"Found {len(false_negatives):,} false negatives\n")

    # Analyze common words
    print("Analyzing most common words in missed toxic messages...")
    common_words = find_common_words(false_negatives, current_words, top_n=50)

    print("\n=== Top 50 Most Common Words in Missed Toxic Messages ===")
    print("(Potential candidates to add to profanity list)\n")

    for i, (word, count) in enumerate(common_words, 1):
        print(f"{i:2}. {word:20} ({count:,} occurrences)")

    print("\n=== Sample False Negative Messages ===")
    print("(Toxic messages our detector missed)\n")
    for i, msg in enumerate(false_negatives[:20], 1):
        print(f"{i:2}. {msg}")

    print(f"\n... and {len(false_negatives) - 20:,} more")

if __name__ == "__main__":
    main()
