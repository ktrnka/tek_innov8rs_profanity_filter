#!/bin/bash
#
# Download datasets for profanity filter training
#
# Usage: cd data && bash download.sh
#

set -e  # Exit on error

echo "Downloading datasets..."

# GameTox dataset
# Note: This is on GitHub, so we can download directly
if [ ! -d "GameTox" ]; then
    echo "Downloading GameTox dataset..."
    git clone https://github.com/shucoll/GameTox.git
    echo "✓ GameTox downloaded"
else
    echo "✓ GameTox already exists"
fi

# Reddit Usernames dataset
# Note: This requires Kaggle credentials via API token
REDDIT_FILE="reddit-usernames.zip"
if [ ! -f "$REDDIT_FILE" ]; then
    echo "Downloading Reddit Usernames dataset..."
    echo "This requires Kaggle credentials."
    echo "Make sure you have a Kaggle account and API token in ~/.kaggle/kaggle.json"
    echo ""
    
    curl -L -o reddit-usernames.zip \
      https://www.kaggle.com/api/v1/datasets/download/colinmorris/reddit-usernames
    
    if [ -f "$REDDIT_FILE" ]; then
        echo "✓ Reddit dataset downloaded"
        # Unzip the dataset
        unzip -q reddit-usernames.zip
        echo "✓ Reddit dataset extracted"
    else
        echo "✗ Download failed. Make sure you have Kaggle credentials set up."
        echo "  1. Create account at https://www.kaggle.com/"
        echo "  2. Go to Account settings > API > Create New Token"
        echo "  3. Place kaggle.json in ~/.kaggle/"
    fi
else
    echo "✓ Reddit dataset already exists"
fi

echo ""
echo "Download complete!"
echo "GameTox data is in: GameTox/"
