#!/usr/bin/env python3
"""Test multi-class classification (4 categories) using prompt V5."""

import os
import csv
import pickle
from dotenv import load_dotenv
from llm_detector import LLMProfanityDetector

load_dotenv()


def load_messages_multiclass(csv_file):
    """Load messages and map labels to 4 classes.

    Original GameTox labels:
    - 0: Non-Toxic → 0 (clean)
    - 1: Insults and Flaming → 1 (insult)
    - 2: Other Offensive Texts → 2 (offensive)
    - 3: Hate and Harassment → 3 (hate_speech)
    - 4: Threats → 3 (hate_speech) - collapsed
    - 5: Extremism → 3 (hate_speech) - collapsed
    """
    messages = []
    labels = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            messages.append(row['message'])
            label = int(float(row['label']))
            # Collapse labels 3,4,5 into 3 (hate_speech)
            if label >= 3:
                label = 3
            labels.append(label)
    return messages, labels


def calculate_multiclass_metrics(y_true, y_pred, num_classes=4):
    """Calculate per-class metrics and confusion matrix."""
    # Confusion matrix
    confusion = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    # Per-class metrics
    class_names = ["clean", "insult", "offensive", "hate_speech"]
    metrics = {}

    for i in range(num_classes):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(num_classes) if j != i)
        fn = sum(confusion[i][j] for j in range(num_classes) if j != i)
        tn = sum(confusion[j][k] for j in range(num_classes) for k in range(num_classes)
                 if j != i and k != i)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[class_names[i]] = {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn  # Number of true examples
        }

    # Overall accuracy
    total_correct = sum(confusion[i][i] for i in range(num_classes))
    total = sum(sum(row) for row in confusion)
    accuracy = total_correct / total if total > 0 else 0

    # Macro-averaged F1 (average F1 across all classes)
    macro_f1 = sum(metrics[c]['f1'] for c in class_names) / num_classes

    # Weighted F1 (weighted by support)
    total_support = sum(metrics[c]['support'] for c in class_names)
    weighted_f1 = sum(metrics[c]['f1'] * metrics[c]['support'] for c in class_names) / total_support if total_support > 0 else 0

    return {
        'confusion_matrix': confusion,
        'per_class': metrics,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


def print_results(results):
    """Pretty print multi-class results."""
    print("\n" + "="*70)
    print("  MULTI-CLASS CLASSIFICATION RESULTS")
    print("="*70)

    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"  Macro F1:    {results['macro_f1']:.3f}")
    print(f"  Weighted F1: {results['weighted_f1']:.3f}")

    # Per-class metrics
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 70)

    class_names = ["clean", "insult", "offensive", "hate_speech"]
    for class_name in class_names:
        m = results['per_class'][class_name]
        print(f"{class_name:<15} {m['support']:>8} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"{'':>15} {'Pred:':>8} {'clean':>8} {'insult':>8} {'offensive':>8} {'hate':>8}")
    print("-" * 70)
    cm = results['confusion_matrix']
    for i, class_name in enumerate(class_names):
        row_label = f"True: {class_name}"
        values = [cm[i][j] for j in range(4)]
        print(f"{row_label:>23} {values[0]:>8} {values[1]:>8} {values[2]:>8} {values[3]:>8}")


def main():
    print("="*70)
    print("  MULTI-CLASS CLASSIFICATION TEST")
    print("="*70)
    print("\nTesting 4-class classification: clean / insult / offensive / hate_speech")
    print("Using Grok + Prompt V5 (multi-class context-aware)")
    print()

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    # Load messages
    messages, labels = load_messages_multiclass('../data/test_subset_200_stratified.csv')
    print(f"✓ Loaded {len(messages)} messages")

    # Show label distribution
    from collections import Counter
    counts = Counter(labels)
    class_names = ["clean", "insult", "offensive", "hate_speech"]
    print(f"\nLabel distribution:")
    for i in range(4):
        pct = counts[i] / len(labels) * 100 if i in counts else 0
        count = counts[i] if i in counts else 0
        print(f"  {class_names[i]:<12}: {count:>3} ({pct:>5.1f}%)")
    print()

    # Test with Grok + Prompt V5
    model = "x-ai/grok-4.1-fast"
    print(f"Model: {model}")
    print(f"Prompt: V5 (Multi-class context-aware)")
    print(f"Batch size: 10")
    print()

    detector = LLMProfanityDetector(model, api_key)

    # Run predictions
    predictions = detector.predict_batch(
        messages,
        show_progress=True,
        prompt_version=5,  # Multi-class prompt
        use_batch_api=True,
        batch_size=10,
        enable_fallback=False
    )

    if predictions is None:
        print("\n❌ Multi-class test failed")
        return

    # Calculate metrics
    results = calculate_multiclass_metrics(labels, predictions, num_classes=4)

    # Print results
    print_results(results)

    # Save results
    with open('multiclass_results.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'prompt_version': 5,
            'results': results,
            'predictions': predictions,
            'labels': labels
        }, f)
    print("\n💾 Results saved to multiclass_results.pkl")

    print("\n" + "="*70)
    print("✓ Multi-class classification test complete!")
    print("="*70)


if __name__ == '__main__':
    main()
