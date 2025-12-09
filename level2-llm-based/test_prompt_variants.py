#!/usr/bin/env python3
"""Test all 4 prompt variants across 3 models to find optimal prompt."""

import os
import csv
import pickle
from dotenv import load_dotenv
from llm_detector import LLMProfanityDetector

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


def calculate_metrics(y_true, y_pred):
    """Calculate TP, FP, TN, FN and metrics."""
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0.0 and pred == 0)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1.0 and pred == 0)

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def test_prompt_variant(model_name, messages, labels, api_key, prompt_version, batch_size=10):
    """Test a single prompt variant on all messages."""
    prompt_names = {
        1: "V1: Concise rule-based",
        2: "V2: Gaming examples",
        3: "V3: Context-aware",
        4: "V4: Conservative"
    }

    print(f"\n{'='*70}")
    print(f"  {model_name} - {prompt_names[prompt_version]}")
    print(f"{'='*70}\n")

    detector = LLMProfanityDetector(model_name, api_key)

    # Run predictions using specified prompt version
    predictions = detector.predict_batch(
        messages,
        show_progress=True,
        prompt_version=prompt_version,
        use_batch_api=True,
        batch_size=batch_size,
        enable_fallback=False
    )

    if predictions is None:
        print(f"\n❌ Test failed for {prompt_names[prompt_version]}")
        return None

    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)

    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1-score:  {metrics['f1']:.3f}")
    print(f"\n  TP: {metrics['tp']:>3}  FP: {metrics['fp']:>3}")
    print(f"  FN: {metrics['fn']:>3}  TN: {metrics['tn']:>3}\n")

    return metrics


def main():
    print("="*70)
    print("  PROMPT VARIANT TESTING")
    print("="*70)
    print("\nTesting 4 prompt variants × 3 models = 12 combinations")
    print("Dataset: 200 stratified messages (same as Level 2 baseline)")
    print()

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    # Load messages
    messages, labels = load_messages('../data/test_subset_200_stratified.csv')
    print(f"✓ Loaded {len(messages)} messages")
    toxic_count = sum(1 for l in labels if l == 1.0)
    print(f"   Toxic: {toxic_count} ({toxic_count/len(labels)*100:.1f}%)")
    print(f"   Clean: {len(labels) - toxic_count} ({(len(labels) - toxic_count)/len(labels)*100:.1f}%)")
    print()

    # Models to test
    models = [
        "openai/gpt-oss-20b:free",
        "x-ai/grok-4.1-fast",
        "meta-llama/llama-3.3-70b-instruct:free"
    ]

    # Prompt versions to test
    prompt_versions = [1, 2, 3, 4]

    # Results storage
    all_results = {}

    # Test each model with each prompt
    for model in models:
        all_results[model] = {}

        for prompt_version in prompt_versions:
            metrics = test_prompt_variant(
                model, messages, labels, api_key,
                prompt_version, batch_size=10
            )

            if metrics:
                all_results[model][f"prompt_v{prompt_version}"] = metrics

                # Save progress after each test
                with open('prompt_variant_results.pkl', 'wb') as f:
                    pickle.dump(all_results, f)
                print("💾 Progress saved to prompt_variant_results.pkl")

    # Print summary comparison
    print("\n" + "="*70)
    print("  SUMMARY: Best F1-Score for Each Model")
    print("="*70)

    for model in models:
        print(f"\n{model}:")
        if model in all_results and all_results[model]:
            sorted_prompts = sorted(
                all_results[model].items(),
                key=lambda x: x[1]['f1'],
                reverse=True
            )
            for prompt_name, metrics in sorted_prompts:
                print(f"  {prompt_name}: F1={metrics['f1']:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")

    print("\n" + "="*70)
    print("✓ Prompt variant testing complete!")
    print("="*70)


if __name__ == '__main__':
    main()
