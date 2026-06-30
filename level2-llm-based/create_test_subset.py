#!/usr/bin/env python3
"""
Create the standard test subset of 4,800 messages from GameTox.
This subset will be used for testing all levels (1, 2, 3, 4) to ensure fair comparison.
"""

import csv
from pathlib import Path


def create_test_subset():
    """Create test_subset_4800.csv from GameTox dataset."""

    # Paths
    gametox_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"
    output_path = Path(__file__).parent.parent / "data" / "test_subset_4800.csv"

    if not gametox_path.exists():
        print(f"ERROR: GameTox dataset not found at {gametox_path}")
        return

    if output_path.exists():
        print(f"Test subset already exists at {output_path}")
        return

    print(f"Creating test subset from {gametox_path}...")

    # Load first 4,800 valid messages
    messages = []
    with open(gametox_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if len(messages) >= 4800:
                break
            if not row['label'] or row['label'].strip() == '':
                continue
            messages.append({
                'index': len(messages),
                'message': row['message'],
                'label': row['label']
            })

    # Save subset
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'message', 'label'])
        writer.writeheader()
        writer.writerows(messages)

    print(f"\n✓ Created test subset: {output_path}")
    print(f"  Messages: {len(messages)}")
    print(f"  This subset will be used for all levels (1, 2, 3, 4)")


if __name__ == "__main__":
    create_test_subset()
