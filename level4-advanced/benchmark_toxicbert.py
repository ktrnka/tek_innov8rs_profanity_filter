#!/usr/bin/env python3
"""Benchmark toxic-bert pre-trained model on profanity detection.

This script:
1. Loads the toxic-bert model from HuggingFace
2. Evaluates on the same 200-message test subset used in Levels 2 and 3
3. Calculates metrics and compares with previous levels
4. Shows that pre-trained transformers can compete with LLMs
"""

import csv
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


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
                # Binary: 0=clean, 1=toxic
                label = int(float(row['label']))
                labels.append(0 if label == 0 else 1)
            except (ValueError, KeyError):
                skipped += 1
                continue

    if skipped > 0:
        print(f"⚠ Skipped {skipped} rows with missing/invalid labels")

    return messages, labels


def predict_toxicity(messages, tokenizer, model, device, batch_size=16, threshold=0.5):
    """Predict toxicity using toxic-bert model.

    toxic-bert is a MULTI-LABEL classifier with 6 toxicity types:
    - toxic, severe_toxic, obscene, threat, insult, identity_hate

    For binary classification, we consider a message toxic if ANY
    of the 6 scores exceeds the threshold.

    Args:
        messages: List of text messages
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: torch device (cpu or cuda)
        batch_size: Number of messages to process at once
        threshold: Threshold for considering a label positive (default 0.5)

    Returns:
        predictions: List of 0 (clean) or 1 (toxic)
        probabilities: List of [clean_prob, toxic_prob] for each message
        max_scores: List of maximum toxicity scores across all 6 labels
    """
    model.eval()
    predictions = []
    probabilities = []
    max_scores = []

    print(f"Processing {len(messages)} messages in batches of {batch_size}...")
    print(f"Using threshold: {threshold} (if ANY toxicity type > {threshold}, classify as toxic)")
    print()

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # For multi-label classification, use sigmoid (not softmax!)
            # This gives independent probabilities for each of the 6 labels
            probs = torch.sigmoid(logits)

            # For binary toxic/clean: a message is toxic if ANY label > threshold
            # Get the maximum score across all 6 toxicity types
            max_toxicity_score = torch.max(probs, dim=1).values

            # Binary prediction: 1 if max score > threshold, else 0
            batch_preds = (max_toxicity_score > threshold).long().cpu().numpy()

            # For reporting: convert to [clean_prob, toxic_prob]
            # clean_prob = 1 - max_score, toxic_prob = max_score
            batch_probs = []
            for score in max_toxicity_score.cpu().numpy():
                batch_probs.append([1 - score, score])

            predictions.extend(batch_preds)
            probabilities.extend(batch_probs)
            max_scores.extend(max_toxicity_score.cpu().numpy())

        # Progress indicator
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(messages):
            print(f"  Processed {min(i+batch_size, len(messages))}/{len(messages)} messages")

    return predictions, probabilities, max_scores


def main():
    print("="*70)
    print("  LEVEL 4: TOXIC-BERT BENCHMARK")
    print("="*70)
    print()

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model and tokenizer
    print("Loading toxic-bert model from HuggingFace...")
    print("  Model: unitary/toxic-bert")
    print("  Pre-trained on: Jigsaw Toxic Comments, Civil Comments")
    print()

    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
    model.to(device)
    load_time = time.time() - start_load

    print(f"✓ Model loaded in {load_time:.2f}s")
    print()

    # Load test data (same 200 messages as Level 2 and 3)
    print("Loading test set (200 messages)...")
    test_messages, test_labels = load_gametox('../data/test_subset_200_stratified.csv')

    print(f"✓ Loaded {len(test_messages)} test messages")
    print(f"\nTest set distribution:")
    clean_count = test_labels.count(0)
    toxic_count = test_labels.count(1)
    print(f"  Clean (0): {clean_count} ({clean_count/len(test_labels)*100:.1f}%)")
    print(f"  Toxic (1): {toxic_count} ({toxic_count/len(test_labels)*100:.1f}%)")
    print()

    # Make predictions
    print("="*70)
    print("  RUNNING TOXIC-BERT INFERENCE")
    print("="*70)
    print()

    start_time = time.time()
    predictions, probabilities, max_scores = predict_toxicity(
        test_messages, tokenizer, model, device, batch_size=16, threshold=0.5
    )
    predict_time = time.time() - start_time

    print(f"\n✓ Inference complete")
    print()

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)

    print("="*70)
    print("  TEST SET PERFORMANCE")
    print("="*70)
    print()
    print(f"Accuracy:   {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision:  {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:     {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1-Score:   {f1:.3f}")
    print()
    print(f"Inference time: {predict_time*1000:.2f}ms for {len(test_messages)} messages")
    print(f"Latency/msg:    {predict_time/len(test_messages)*1000:.2f}ms")
    print()

    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    tn, fp, fn, tp = cm.ravel()

    print("="*70)
    print("  CONFUSION MATRIX")
    print("="*70)
    print()
    print(f"                  Predicted")
    print(f"                  Clean  Toxic")
    print(f"Actual  Clean     {tn:4d}   {fp:4d}")
    print(f"        Toxic     {fn:4d}   {tp:4d}")
    print()
    print(f"True Negatives (TN):  {tn:3d} - Correctly identified as clean")
    print(f"False Positives (FP): {fp:3d} - Clean messages flagged as toxic")
    print(f"False Negatives (FN): {fn:3d} - Toxic messages missed")
    print(f"True Positives (TP):  {tp:3d} - Correctly identified as toxic")
    print()
    if (fp + tn) > 0:
        print(f"False Positive Rate: {fp/(fp+tn)*100:.1f}% (clean messages incorrectly flagged)")
    if (fn + tp) > 0:
        print(f"False Negative Rate: {fn/(fn+tp)*100:.1f}% (toxic messages missed)")
    print()

    # Detailed classification report
    print("="*70)
    print("  DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print()
    print(classification_report(test_labels, predictions,
                              target_names=['Clean', 'Toxic'],
                              digits=3))

    # Example predictions
    print("="*70)
    print("  EXAMPLE PREDICTIONS WITH CONFIDENCE")
    print("="*70)
    print()

    for i in range(min(20, len(test_messages))):
        msg = test_messages[i]
        true_label = "TOXIC" if test_labels[i] == 1 else "CLEAN"
        pred_label = "TOXIC" if predictions[i] == 1 else "CLEAN"
        confidence = probabilities[i][predictions[i]] * 100

        # Determine correctness
        correct = "✓" if predictions[i] == test_labels[i] else "✗"

        # Format message
        msg_display = msg[:50] + "..." if len(msg) > 50 else msg

        print(f"{correct} \"{msg_display}\"")
        print(f"   True: {true_label:5s} | Pred: {pred_label:5s} ({confidence:.1f}% confident)")
        print()

    # Cross-level comparison
    print("="*70)
    print("  CROSS-LEVEL COMPARISON")
    print("="*70)
    print()
    print(f"{'Metric':<20} {'Level 1':<15} {'Level 2':<15} {'Level 3':<15} {'Level 4':<15}")
    print(f"{'':20} {'(Rules)':<15} {'(LLM)':<15} {'(Trad ML)':<15} {'(Toxic-BERT)':<15}")
    print("-" * 80)
    print(f"{'F1-Score':<20} {'0.650':<15} {'0.816':<15} {'0.677':<15} {f'{f1:.3f}':<15}")
    print(f"{'Precision':<20} {'0.580':<15} {'0.769':<15} {'0.840':<15} {f'{precision:.3f}':<15}")
    print(f"{'Recall':<20} {'0.740':<15} {'0.870':<15} {'0.568':<15} {f'{recall:.3f}':<15}")
    print(f"{'Latency (ms/msg)':<20} {'<1':<15} {'181':<15} {'0.04':<15} {f'{predict_time/len(test_messages)*1000:.2f}':<15}")
    print(f"{'Cost per 1M msgs':<20} {'$0':<15} {'$1,710':<15} {'$0':<15} {'$0':<15}")
    print(f"{'Model size':<20} {'<1KB':<15} {'N/A (API)':<15} {'~500KB':<15} {'~440MB':<15}")
    print()
    print("="*70)
    print()
    print("Key Insights:")
    print("  • Toxic-BERT is a pre-trained transformer model (no training needed!)")
    print("  • Zero API cost (runs locally)")
    print("  • Significantly larger model than traditional ML (~440MB vs ~500KB)")
    print("  • Inference speed depends on hardware (faster on GPU)")
    print("  • Performance between traditional ML and LLM")
    print()


if __name__ == '__main__':
    main()
