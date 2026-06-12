#!/usr/bin/env python3
"""Train a traditional ML classifier for profanity detection.

This script:
1. Loads the full GameTox dataset for training
2. Trains a TF-IDF + LogisticRegression pipeline
3. Saves the trained model to disk
4. Shows what the model learned (feature weights)
"""

import csv
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd


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
                # Binary: 0=clean, 1=toxic (any non-zero label is toxic)
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
    print("  LEVEL 3: TRADITIONAL ML CLASSIFIER - TRAINING")
    print("="*70)
    print()

    # Load training data
    print("Loading GameTox dataset...")
    train_messages, train_labels = load_gametox('../data/GameTox/GameTox.csv')

    print(f"✓ Loaded {len(train_messages):,} messages")
    print(f"\nClass distribution:")
    clean_count = train_labels.count(0)
    toxic_count = train_labels.count(1)
    print(f"  Clean (0): {clean_count:,} ({clean_count/len(train_labels)*100:.1f}%)")
    print(f"  Toxic (1): {toxic_count:,} ({toxic_count/len(train_labels)*100:.1f}%)")
    print()

    # Create pipeline with recommended settings
    print("Building ML pipeline...")
    print("\nTF-IDF Settings:")
    print("  - sublinear_tf: True (reduces impact of very frequent words)")
    print("  - max_df: 0.5 (ignore words in >50% of docs)")
    print("  - min_df: 5 (ignore rare words in <5 docs)")
    print("  - stop_words: 'english' (remove common words)")
    print("  - ngram_range: (1, 1) (single words only)")
    print("\nLogistic Regression Settings:")
    print("  - C: 5 (regularization strength)")
    print("  - max_iter: 1000")
    print()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.5,
            min_df=5,
            stop_words='english',
            ngram_range=(1, 1)
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=5,
            verbose=1  # Show training progress
        ))
    ])

    # Train the model
    print("Training model...")
    start_time = time.time()
    pipeline.fit(train_messages, train_labels)
    train_time = time.time() - start_time

    print(f"✓ Training complete in {train_time:.2f}s")
    print()

    # Show vocabulary statistics
    vocab_size = len(pipeline.named_steps['tfidf'].vocabulary_)
    print(f"Vocabulary statistics:")
    print(f"  - Features after TF-IDF filtering: {vocab_size:,}")
    print()

    # Show top features (what the model learned)
    print("="*70)
    print("  WHAT DID THE MODEL LEARN?")
    print("="*70)
    print()

    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    coefficients = pipeline.named_steps['clf'].coef_[0]

    # Create DataFrame for sorting
    coef_df = pd.DataFrame({
        'word': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)

    # Top words for TOXIC
    print("TOP 20 WORDS INDICATING TOXIC CONTENT:")
    print(f"{'Rank':<6} {'Word':<20} {'Weight':>10}")
    print("-" * 70)
    for i, (_, row) in enumerate(coef_df.head(20).iterrows(), 1):
        print(f"{i:<6} {row['word']:<20} {row['coefficient']:>10.3f}")

    print()
    print("TOP 20 WORDS INDICATING CLEAN CONTENT:")
    print(f"{'Rank':<6} {'Word':<20} {'Weight':>10}")
    print("-" * 70)
    for i, (_, row) in enumerate(coef_df.tail(20).iloc[::-1].iterrows(), 1):
        print(f"{i:<6} {row['word']:<20} {row['coefficient']:>10.3f}")

    print()

    # Save the model
    model_path = 'profanity_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    print("="*70)
    print(f"✓ Model saved to {model_path}")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run evaluate_classifier.py to test on the 200-message subset")
    print("  2. Compare results with Level 1 (rule-based) and Level 2 (LLM)")
    print()


if __name__ == '__main__':
    main()
