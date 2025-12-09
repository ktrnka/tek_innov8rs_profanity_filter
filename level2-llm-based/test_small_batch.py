#!/usr/bin/env python3
"""
Quick test: Verify batch classification works with a small sample.
Tests a paid model on just 10 messages to confirm the approach before full run.
"""

import csv
import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Load first 10 messages
data_path = Path(__file__).parent.parent / "data" / "test_subset_4800.csv"
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    messages = [row['message'] for i, row in enumerate(reader) if i < 10]

print(f"Testing with {len(messages)} messages")
print(f"Messages: {messages[:3]}...")

# Build batch prompt
numbered_messages = "\n".join([f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)])

batch_prompt = f"""Classify each of the following {len(messages)} messages as either "toxic" or "clean".

A message is TOXIC if it contains:
- Profanity or vulgar language
- Insults or personal attacks
- Hate speech or discriminatory language

A message is CLEAN otherwise.

Messages:
{numbered_messages}

Respond with ONLY a JSON array of {len(messages)} classifications in the same order.
Use exactly "toxic" or "clean" for each entry (lowercase).

Example format: ["toxic", "clean", "toxic", "clean", ...]
"""

# Test with a reliable paid model (GPT-3.5)
model = "openai/gpt-3.5-turbo"

print(f"\nTesting model: {model}")
print(f"Prompt length: {len(batch_prompt)} chars")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": batch_prompt}],
    "temperature": 0.0,
    "max_tokens": 500
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json=payload,
    timeout=30
)

result = response.json()
print(f"\nStatus: {response.status_code}")

if response.status_code == 200:
    answer = result['choices'][0]['message']['content'].strip()
    print(f"\nRaw response:\n{answer}\n")

    # Try to parse
    try:
        classifications = json.loads(answer)
        print(f"✓ Parsed successfully!")
        print(f"  Expected: {len(messages)} items")
        print(f"  Got: {len(classifications)} items")
        print(f"  Classifications: {classifications}")

        # Check token usage
        if 'usage' in result:
            usage = result['usage']
            print(f"\nToken usage:")
            print(f"  Input: {usage.get('prompt_tokens', 0)}")
            print(f"  Output: {usage.get('completion_tokens', 0)}")
            print(f"  Total: {usage.get('total_tokens', 0)}")
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed: {e}")
else:
    print(f"Error: {result}")
