#!/usr/bin/env python3
"""
Download external datasets for comprehensive generalization testing.

Strategy: Test TWO types of generalization
A. Cross-Domain (gaming → non-gaming): Did model learn general toxicity?
B. Within-Domain (World of Tanks → Dota 2): Did model learn gaming toxicity?

Datasets:
1. HateXplain (Twitter/Gab) - Cross-domain, social media
2. Jigsaw Toxic Comments (Wikipedia) - Cross-domain, formal discussion
3. GOSU.ai Dota 2 - Within-domain, different game
"""

import os
import pandas as pd
from datasets import load_dataset
import urllib.request
import json
from sklearn.model_selection import train_test_split

# Create external datasets directory
data_dir = "../data/external_datasets"
os.makedirs(data_dir, exist_ok=True)

print("=" * 70)
print("DOWNLOADING EXTERNAL DATASETS FOR GENERALIZATION TESTING")
print("=" * 70)
print("\nStrategy:")
print("  A. Cross-Domain: Gaming → Non-Gaming (general toxicity?)")
print("  B. Within-Domain: World of Tanks → Dota 2 (gaming-specific?)")
print("=" * 70)

# ============================================================================
# 1. HateXplain Dataset (Cross-Domain: Social Media)
# ============================================================================
print("\n[1/3] Downloading HateXplain (Twitter/Gab)...")
print("  Purpose: Test cross-domain generalization to social media")
try:
    # Load from HuggingFace
    hatexplain = load_dataset("hatexplain", split="train")

    # Convert to pandas DataFrame
    hatexplain_df = hatexplain.to_pandas()

    # HateXplain has labels: 0=normal, 1=hatespeech, 2=offensive
    # Convert to binary: 0=clean, 1/2=toxic
    hatexplain_df['is_toxic'] = hatexplain_df['label'].apply(lambda x: 1 if x in [1, 2] else 0)
    hatexplain_df['text'] = hatexplain_df['post_tokens'].apply(lambda x: ' '.join(x))

    # Save to CSV
    hatexplain_path = os.path.join(data_dir, "hatexplain.csv")
    hatexplain_df[['text', 'is_toxic']].to_csv(hatexplain_path, index=False)

    print(f"  ✓ Downloaded: {len(hatexplain_df)} samples")
    toxic_count = hatexplain_df['is_toxic'].sum()
    clean_count = len(hatexplain_df) - toxic_count
    toxic_pct = (toxic_count / len(hatexplain_df)) * 100
    print(f"  ✓ Distribution: {toxic_count} toxic ({toxic_pct:.1f}%), {clean_count} clean ({100-toxic_pct:.1f}%)")
    print(f"  ✓ Sample: '{hatexplain_df.iloc[0]['text'][:80]}...'")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. Jigsaw Toxic Comment Dataset (Cross-Domain: Wikipedia)
# ============================================================================
print("\n[2/3] Downloading Jigsaw Toxic Comments (Wikipedia)...")
print("  Purpose: Test cross-domain generalization to formal discussion")
print("  Note: Large dataset - sampling 5K for efficiency")
try:
    # Load from HuggingFace (civil_comments is more available than jigsaw)
    # Try civil_comments first as it's similar and more accessible
    print("  Attempting to load civil_comments dataset...")
    jigsaw = load_dataset("civil_comments", split="train", trust_remote_code=True)

    # Convert to pandas
    jigsaw_df = jigsaw.to_pandas()

    # civil_comments has 'toxicity' column with float values 0-1
    jigsaw_df['is_toxic'] = jigsaw_df['toxicity'].apply(lambda x: 1 if x >= 0.5 else 0)

    # Take a stratified sample (5K samples to reduce size)
    jigsaw_sample, _ = train_test_split(
        jigsaw_df[['text', 'is_toxic']],
        train_size=5000,
        stratify=jigsaw_df['is_toxic'],
        random_state=42
    )

    # Save to CSV
    jigsaw_path = os.path.join(data_dir, "civil_comments_sample.csv")
    jigsaw_sample.to_csv(jigsaw_path, index=False)

    print(f"  ✓ Downloaded: {len(jigsaw_sample)} samples (stratified sample)")
    toxic_count = jigsaw_sample['is_toxic'].sum()
    clean_count = len(jigsaw_sample) - toxic_count
    toxic_pct = (toxic_count / len(jigsaw_sample)) * 100
    print(f"  ✓ Distribution: {toxic_count} toxic ({toxic_pct:.1f}%), {clean_count} clean ({100-toxic_pct:.1f}%)")
    print(f"  ✓ Sample: '{jigsaw_sample.iloc[0]['text'][:80]}...'")

except Exception as e:
    print(f"  ✗ Error: {e}")
    print("  → Trying alternative: jigsaw_toxicity_pred...")

    try:
        jigsaw = load_dataset("google/jigsaw_toxicity_pred", split="train")
        jigsaw_df = jigsaw.to_pandas()
        jigsaw_df['is_toxic'] = jigsaw_df['toxicity'].apply(lambda x: 1 if x >= 0.5 else 0)
        jigsaw_df = jigsaw_df.rename(columns={'comment_text': 'text'})

        jigsaw_sample, _ = train_test_split(
            jigsaw_df[['text', 'is_toxic']],
            train_size=5000,
            stratify=jigsaw_df['is_toxic'],
            random_state=42
        )

        jigsaw_path = os.path.join(data_dir, "jigsaw_sample.csv")
        jigsaw_sample.to_csv(jigsaw_path, index=False)

        print(f"  ✓ Downloaded: {len(jigsaw_sample)} samples")
        toxic_count = jigsaw_sample['is_toxic'].sum()
        print(f"  ✓ Distribution: {toxic_count} toxic, {len(jigsaw_sample) - toxic_count} clean")

    except Exception as e2:
        print(f"  ✗ Both attempts failed: {e2}")
        print("  → Manual download needed from Kaggle")

# ============================================================================
# 3. GOSU.ai Dota 2 Dataset (Within-Domain: Gaming)
# ============================================================================
print("\n[3/3] Downloading GOSU.ai Dota 2 dataset...")
print("  Purpose: Test within-domain generalization to different game")
try:
    # GOSU.ai Dota 2 dataset on HuggingFace or direct download
    # Check if available on HuggingFace first
    print("  Searching for GOSU.ai dataset...")

    # Try loading from HuggingFace (if available)
    # If not, we'll need to download from the source
    # For now, let's check if it's available

    # The dataset might be at: https://huggingface.co/datasets/gosu-ai/dota2-toxicity
    # Or we need to find the original source from the paper

    print("  → Checking HuggingFace for gosu-ai/dota2...")

    try:
        dota2 = load_dataset("gosu-ai/dota2", split="train")
        dota2_df = dota2.to_pandas()

        # GOSU.ai labels: 0=clean, 1=mild toxicity, 2=strong toxicity
        # Convert to binary: 0=clean, 1/2=toxic
        dota2_df['is_toxic'] = dota2_df['label'].apply(lambda x: 1 if x > 0 else 0)

        # Sample if too large
        if len(dota2_df) > 10000:
            dota2_sample, _ = train_test_split(
                dota2_df[['text', 'is_toxic']],
                train_size=10000,
                stratify=dota2_df['is_toxic'],
                random_state=42
            )
        else:
            dota2_sample = dota2_df[['text', 'is_toxic']]

        # Save to CSV
        dota2_path = os.path.join(data_dir, "dota2_gosu.csv")
        dota2_sample.to_csv(dota2_path, index=False)

        print(f"  ✓ Downloaded: {len(dota2_sample)} samples")
        toxic_count = dota2_sample['is_toxic'].sum()
        print(f"  ✓ Distribution: {toxic_count} toxic, {len(dota2_sample) - toxic_count} clean")

    except:
        print("  ✗ Not found on HuggingFace with that name")
        print("  → Searching alternative sources...")

        # Try alternative: download from paper's GitHub or Kaggle
        print("  → Dataset may require manual download")
        print("  → Source: https://arxiv.org/html/2510.17924v1")
        print("  → Alternative: Use CONDA (Dota 2) dataset instead")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

# Check what we successfully downloaded
datasets_found = []
if os.path.exists(os.path.join(data_dir, "hatexplain.csv")):
    datasets_found.append("✓ HateXplain (Twitter/Gab)")
if os.path.exists(os.path.join(data_dir, "civil_comments_sample.csv")) or \
   os.path.exists(os.path.join(data_dir, "jigsaw_sample.csv")):
    datasets_found.append("✓ Jigsaw/Civil Comments (Wikipedia)")
if os.path.exists(os.path.join(data_dir, "dota2_gosu.csv")):
    datasets_found.append("✓ GOSU.ai Dota 2 (Gaming)")

print(f"\nSuccessfully downloaded: {len(datasets_found)}/3 datasets")
for ds in datasets_found:
    print(f"  {ds}")

print(f"\nLocation: {data_dir}/")
print("\nNext step: Create evaluate_external_datasets.py to test all 4 approaches")
