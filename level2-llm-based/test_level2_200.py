#!/usr/bin/env python3
"""Test Level 2 (LLM-based) on 200-message stratified sample for fair comparison with Level 1."""

import os
import time
import csv
import json
import pickle
from dotenv import load_dotenv
import requests

load_dotenv()

def load_messages(csv_file):
    """Load messages and labels from CSV."""
    messages = []
    labels = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row['message'])
            labels.append(float(row['label']))

    return messages, labels

def classify_batch(model_name, messages_batch, api_key, request_delay=1.0):
    """
    Classify a batch of messages using OpenRouter API.

    Returns tuple: (classifications, latency, input_tokens, output_tokens, error_msg)
    """
    # Create the prompt for batch classification
    messages_text = '\n'.join([f'{i+1}. "{msg}"' for i, msg in enumerate(messages_batch)])

    prompt = f"""Classify each of the {len(messages_batch)} messages below as either "toxic" or "clean".

Messages:
{messages_text}

Respond with ONLY a JSON array of {len(messages_batch)} classifications, like: ["toxic", "clean", "toxic", ...]
Do not include any explanation, just the JSON array."""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        latency = time.time() - start_time

        response.raise_for_status()
        data = response.json()

        # Extract response content
        content = data['choices'][0]['message']['content'].strip()

        # Extract token usage
        usage = data.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)

        # Parse JSON array from response
        # Sometimes models wrap it in markdown code blocks
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
            content = content.strip()

        classifications = json.loads(content)

        # Validate we got the right number of classifications
        if len(classifications) != len(messages_batch):
            return None, latency, input_tokens, output_tokens, f"Expected {len(messages_batch)} classifications, got {len(classifications)}"

        # Wait for rate limiting
        time.sleep(request_delay)

        return classifications, latency, input_tokens, output_tokens, None

    except Exception as e:
        return None, 0, 0, 0, str(e)

def calculate_metrics(y_true, y_pred):
    """Calculate TP, FP, TN, FN and metrics."""
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 'toxic')
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 'toxic')
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 'clean')
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 'clean')

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flagged': tp + fp
    }

def test_model(model_name, messages, labels, api_key, batch_size=10):
    """Test a single model on all messages."""
    print(f"\n{'─'*70}")
    print(f"Testing: {model_name}")
    print(f"{'─'*70}\n")

    all_predictions = []
    total_latency = 0
    total_input_tokens = 0
    total_output_tokens = 0
    num_batches = (len(messages) + batch_size - 1) // batch_size

    print(f"   Processing {len(messages)} messages in {num_batches} batches of ~{batch_size}...\n")

    for i in range(0, len(messages), batch_size):
        batch_num = i // batch_size + 1
        batch = messages[i:i + batch_size]
        progress = min((i + batch_size) / len(messages) * 100, 100)
        total_so_far = min(i + batch_size, len(messages))

        print(f"   Batch {batch_num}/{num_batches} ({len(batch)} messages) [{progress:.0f}% - {total_so_far}/{len(messages)} total]")
        print(f"\n   Sending batch request ({len(batch)} messages)...\n")

        classifications, latency, input_tok, output_tok, error = classify_batch(
            model_name, batch, api_key, request_delay=1.0
        )

        if error:
            print(f"   ❌ Batch {batch_num} FAILED: {error}")
            return None

        all_predictions.extend(classifications)
        total_latency += latency
        total_input_tokens += input_tok
        total_output_tokens += output_tok

        print(f"   ✓ Batch classified in {latency:.2f}s\n")

    # Calculate metrics
    metrics = calculate_metrics(labels, all_predictions)

    metrics['total_requests'] = num_batches
    metrics['total_latency'] = total_latency
    metrics['avg_latency'] = (total_latency / len(messages)) * 1000  # ms per message
    metrics['total_input_tokens'] = total_input_tokens
    metrics['total_output_tokens'] = total_output_tokens
    metrics['avg_input_tokens'] = total_input_tokens / num_batches
    metrics['avg_output_tokens'] = total_output_tokens / num_batches

    # Print results
    print("="*70)
    print(f"  MODEL: {model_name}")
    print("="*70)
    print(f"Messages evaluated: {len(messages)}")
    print(f"Messages flagged: {metrics['flagged']} ({metrics['flagged']/len(messages)*100:.1f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {metrics['tp']:>3}  |  FP: {metrics['fp']:>3}")
    print(f"  FN: {metrics['fn']:>3}  |  TN: {metrics['tn']:>3}")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1-score:  {metrics['f1']:.3f}")
    print()
    print("Performance:")
    print(f"  Total requests: {metrics['total_requests']}")
    print(f"  Total time: {metrics['total_latency']:.1f}s")
    print(f"  Avg latency: {metrics['avg_latency']:.0f}ms per message")
    print(f"  Avg input tokens: {metrics['avg_input_tokens']:.1f}")
    print(f"  Avg output tokens: {metrics['avg_output_tokens']:.1f}")
    print()

    return metrics, all_predictions

def main():
    print("="*70)
    print("  LEVEL 2: LLM-BASED TESTING (200-message stratified sample)")
    print("="*70)
    print()

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    print("✓ API key found\n")

    # Load messages
    messages, labels = load_messages('../data/test_subset_200_stratified.csv')
    print(f"✓ Loaded {len(messages)} messages")
    toxic_count = sum(1 for l in labels if l == 1.0)
    print(f"   Toxic: {toxic_count} ({toxic_count/len(labels)*100:.1f}%)")
    print(f"   Clean: {len(labels) - toxic_count} ({(len(labels) - toxic_count)/len(labels)*100:.1f}%)")
    print()

    # Test 3 free models
    models = [
        "openai/gpt-oss-20b:free",
        "x-ai/grok-4.1-fast",
        "meta-llama/llama-3.3-70b-instruct:free"
    ]

    results = {}

    for model in models:
        result = test_model(model, messages, labels, api_key, batch_size=10)
        if result:
            metrics, predictions = result
            results[model] = {
                'metrics': metrics,
                'predictions': predictions
            }

            # Save progress
            with open('level2_200_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("💾 Progress saved to level2_200_results.pkl\n")

    print("="*70)
    print("  LEVEL 2 TESTING COMPLETE!")
    print("="*70)
    print()
    print("✓ All 3 free models tested on 200-message sample")
    print("✓ Results saved for comparison with Level 1")
    print()

if __name__ == '__main__':
    main()
