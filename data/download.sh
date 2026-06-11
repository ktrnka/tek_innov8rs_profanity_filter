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
# (github.com/shucoll/GameTox) is now README-only; the data is hosted on a public
# Google Drive folder. We use `uvx` to run gdown without adding a project dependency.
# The folder contains train.csv (index,message,label) plus index_text/index_label splits.
if [ ! -f "GameTox/train.csv" ]; then
    echo "Downloading GameTox dataset from Google Drive..."
    uvx gdown --folder "https://drive.google.com/drive/folders/1HkfwexOpX1S9gRrMeCFMfZJjsBs0hQRu" -O GameTox
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
