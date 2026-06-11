#!/bin/bash
#
# Download datasets for profanity filter training
#
# Usage: cd data && bash download.sh
#

set -e  # Exit on error

echo "Downloading datasets..."

# GameTox dataset (labeled gaming chat — used in Levels 1-3)
# Source: GameTox NAACL 2025 shared task. The original GitHub repo
# (github.com/shucoll/GameTox) is now README-only; the data is hosted in a public
# Google Drive folder:
#   https://drive.google.com/drive/folders/1HkfwexOpX1S9gRrMeCFMfZJjsBs0hQRu
# We download each file directly with curl so there's no extra tooling to install.
# (If these IDs ever stop working, open the folder above and grab the new file IDs.)
# train.csv has the index,message,label columns we use; the index_* files are alternate splits.
if [ ! -f "GameTox/train.csv" ]; then
    echo "Downloading GameTox dataset from Google Drive..."
    mkdir -p GameTox
    # "<filename> <google-drive-file-id>" pairs (portable; no bash-4 associative arrays).
    for entry in \
        "train.csv 1A2UffUVBiCdvL3EHbethWEdjjCeKTLAh" \
        "train_index_text.csv 1WVlDmARNyyTEnUd4BV2aRbfVOMTzELsq" \
        "train_index_label.csv 1caB5kzKzmne03rixioAXt92V2wh7iUIC"; do
        set -- $entry
        echo "  - $1"
        curl -fsSL "https://drive.usercontent.google.com/download?id=$2&export=download" -o "GameTox/$1"
    done
    # Sanity-check we got a real CSV and not an HTML error page.
    if ! head -1 GameTox/train.csv | grep -q "message"; then
        echo "✗ GameTox/train.csv doesn't look right (missing 'message' header)."
        echo "  The Google Drive file IDs may have changed — see the folder URL above."
        exit 1
    fi
    echo "✓ GameTox downloaded to data/GameTox/"
else
    echo "✓ GameTox already exists"
fi

# Reddit Usernames dataset (unlabeled — OPTIONAL, only needed for the Level 4 username exercise)
# Note: This requires Kaggle credentials via API token
REDDIT_FILE="reddit-usernames.zip"
if [ -f "users.csv" ]; then
    echo "✓ Reddit dataset already exists"
elif [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "⚠ Skipping Reddit usernames (optional, Level 4): no ~/.kaggle/kaggle.json found."
    echo "  To enable it later: create a Kaggle account, then Account > API > Create New Token,"
    echo "  and place kaggle.json in ~/.kaggle/. Then re-run this script."
else
    echo "Downloading Reddit Usernames dataset (optional, Level 4)..."
    # -f makes curl fail on HTTP errors; guarded by `if` so a failure doesn't abort the script.
    if curl -fL -o "$REDDIT_FILE" \
        https://www.kaggle.com/api/v1/datasets/download/colinmorris/reddit-usernames \
        && unzip -q -o "$REDDIT_FILE"; then
        echo "✓ Reddit dataset downloaded and extracted"
    else
        echo "⚠ Reddit download failed (check Kaggle credentials). Skipping — only needed for Level 4."
    fi
fi

echo ""
echo "Download complete!"
echo "GameTox data is in: data/GameTox/ (train.csv is the labeled set used in Levels 1-3)"
