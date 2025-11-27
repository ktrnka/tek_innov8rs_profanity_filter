#!/usr/bin/env python3
"""
Test profanity detector on Reddit usernames to identify false positives.
Usernames are usually neutral, so flags are likely false positives.
"""

import csv
import re
from pathlib import Path
from collections import Counter

class RegexProfanityDetector:
    """Simple regex-based profanity detector."""

    def __init__(self, profane_words):
        self.profane_words = profane_words
        pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def is_profane(self, text):
        return bool(self.pattern.search(text))

    def find_matches(self, text):
        """Find which profane words matched."""
        return self.pattern.findall(text)

def load_reddit_usernames(data_path, limit=100000):
    """Load Reddit usernames (sample to avoid loading all 26M)."""
    usernames = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if i >= limit:
                break
            if row:  # Check row is not empty
                usernames.append(row[0])  # Username is first column

    return usernames

def main():
    data_path = Path(__file__).parent.parent / "data" / "reddit-usernames" / "users.csv"

    # Our expanded word list
    profane_words = [
        'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell',
        'idiot', 'idiots', 'wtf', 'fucking', 'useless', 'stupid',
        'retard', 'retards', 'moron', 'morons', 'ffs', 'fck',
        'stfu', 'trash', 'dumb', 'noob', 'noobs', 'bot', 'bots',
        'camper', 'campers', 'camping'
    ]

    print(f"Loading sample of Reddit usernames...")
    usernames = load_reddit_usernames(data_path, limit=100000)
    print(f"Loaded {len(usernames):,} usernames\n")

    # Create detector
    detector = RegexProfanityDetector(profane_words)

    # Test detector
    print("Testing detector on usernames...")
    flagged = []
    for username in usernames:
        if detector.is_profane(username):
            matches = detector.find_matches(username)
            flagged.append((username, matches))

    print(f"Flagged {len(flagged):,} usernames ({100 * len(flagged) / len(usernames):.2f}%)\n")

    # Analyze which words cause most flags
    word_counts = Counter()
    for username, matches in flagged:
        for match in matches:
            word_counts[match.lower()] += 1

    print("=== Words Causing Most Flags ===")
    for word, count in word_counts.most_common(20):
        print(f"{word:15} {count:,} flags")

    # Show sample flagged usernames
    print("\n=== Sample Flagged Usernames ===")
    print("(Likely false positives - these are probably innocent usernames)\n")
    for i, (username, matches) in enumerate(flagged[:50], 1):
        matched_words = ', '.join(set(m.lower() for m in matches))
        print(f"{i:2}. {username:30} (matched: {matched_words})")

    if len(flagged) > 50:
        print(f"\n... and {len(flagged) - 50:,} more")

    # Identify problematic patterns
    print("\n=== Analysis ===")
    print(f"Out of {len(usernames):,} usernames, {len(flagged):,} were flagged.")
    print(f"Flag rate: {100 * len(flagged) / len(usernames):.2f}%")
    print()
    print("Common false positive patterns:")
    print("- Usernames containing 'bot' (could be legitimate bot accounts)")
    print("- Usernames with 'ass' substring (assassin, bass, glass, etc.)")
    print("- Usernames with gaming terms (noob, camper)")
    print()
    print("⚠️  For username filtering, you might want to:")
    print("  1. Remove some words (bot, noob, camper) - too common in usernames")
    print("  2. Add substring checks to avoid 'assassin' → 'ass' matches")
    print("  3. Use a stricter word list for usernames vs. chat messages")

if __name__ == "__main__":
    main()
