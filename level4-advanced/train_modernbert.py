#!/usr/bin/env python3
"""
Fine-tune ModernBERT on GameTox dataset for profanity detection.

This script trains ModernBERT-base for binary toxicity classification
using the full GameTox dataset (53K messages).

Usage:
    python train_modernbert.py --learning_rate 2e-5 --epochs 3 --batch_size 16
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset

print("=" * 70)
print("MODERNBERT FINE-TUNING FOR PROFANITY DETECTION")
print("=" * 70)

# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Fine-tune ModernBERT on GameTox')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate (default: 2e-5)')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs (default: 3)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Training batch size (default: 16)')
parser.add_argument('--max_length', type=int, default=512,
                    help='Maximum sequence length (default: 512)')
parser.add_argument('--output_dir', type=str, default='./modernbert_finetuned',
                    help='Output directory for model')
parser.add_argument('--run_name', type=str, default='run1_baseline',
                    help='Run name for tracking (e.g., run1_baseline, run2_higher_lr)')

args = parser.parse_args()

print(f"\nTraining Configuration:")
print(f"  Learning rate: {args.learning_rate}")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size}")
print(f"  Max length: {args.max_length}")
print(f"  Output dir: {args.output_dir}")
print(f"  Run name: {args.run_name}")

# ============================================================================
# Load and Prepare Data
# ============================================================================

print("\n" + "=" * 70)
print("LOADING GAMETOX DATA")
print("=" * 70)

# Load full GameTox dataset
gametox_path = '../data/GameTox/gametox.csv'
df = pd.read_csv(gametox_path)

print(f"\nLoaded {len(df)} messages from GameTox")
print(f"Columns: {list(df.columns)}")

# GameTox format: 'message' column, 'label' column (0.0 = clean, others = toxic)
# Convert label to binary integer
df['label'] = (df['label'] != 0.0).astype(int)

# Rename message to text for consistency
df = df[['message', 'label']].rename(columns={'message': 'text'})

# Remove any NaN values
df = df.dropna()

print(f"\nClean messages: {(df['label'] == 0).sum()} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"Toxic messages: {(df['label'] == 1).sum()} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")

# Train/validation split (80/20, stratified)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nTrain set: {len(train_df)} messages")
print(f"  Clean: {(train_df['label'] == 0).sum()}, Toxic: {(train_df['label'] == 1).sum()}")
print(f"Validation set: {len(val_df)} messages")
print(f"  Clean: {(val_df['label'] == 0).sum()}, Toxic: {(val_df['label'] == 1).sum()}")

# ============================================================================
# Load Model and Tokenizer
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MODERNBERT MODEL")
print("=" * 70)

model_name = "answerdotai/ModernBERT-base"

print(f"\nLoading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification
    problem_type="single_label_classification"
)

# Move to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

print(f"✓ Model loaded on {device}")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# Tokenize Data
# ============================================================================

print("\n" + "=" * 70)
print("TOKENIZING DATA")
print("=" * 70)

def tokenize_function(examples):
    """Tokenize texts."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=args.max_length,
        return_tensors=None
    )

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(f"\nTokenizing {len(train_dataset)} training examples...")
train_dataset = train_dataset.map(tokenize_function, batched=True)

print(f"Tokenizing {len(val_dataset)} validation examples...")
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print("✓ Tokenization complete")

# ============================================================================
# Define Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """Compute precision, recall, F1, accuracy."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# ============================================================================
# Training Configuration
# ============================================================================

print("\n" + "=" * 70)
print("CONFIGURING TRAINER")
print("=" * 70)

output_dir = f"{args.output_dir}/{args.run_name}"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir=f'{output_dir}/logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    fp16=False,  # Disable mixed precision for MPS compatibility
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    remove_unused_columns=True,
)

print(f"\nTraining arguments:")
print(f"  Output dir: {output_dir}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Warmup ratio: {training_args.warmup_ratio}")
print(f"  Weight decay: {training_args.weight_decay}")

# ============================================================================
# Initialize Trainer
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("\n✓ Trainer initialized")

# ============================================================================
# Train Model
# ============================================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print("\n⚠️  This will take 1-2 hours. Training runs in background.")
print("    You can monitor progress in the terminal.\n")

# Train!
train_result = trainer.train()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nTraining metrics:")
print(f"  Total time: {train_result.metrics['train_runtime']:.1f}s ({train_result.metrics['train_runtime'] / 60:.1f} min)")
print(f"  Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
print(f"  Steps/second: {train_result.metrics['train_steps_per_second']:.2f}")

# ============================================================================
# Evaluate on Validation Set
# ============================================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

eval_results = trainer.evaluate()

print(f"\nValidation metrics:")
print(f"  F1:        {eval_results['eval_f1']:.4f}")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall:    {eval_results['eval_recall']:.4f}")
print(f"  Accuracy:  {eval_results['eval_accuracy']:.4f}")

# ============================================================================
# Save Model
# ============================================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

final_model_path = f"{output_dir}/final_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"\n✓ Model saved to: {final_model_path}")

# Save training results summary
results_summary = {
    'run_name': args.run_name,
    'learning_rate': args.learning_rate,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'val_f1': eval_results['eval_f1'],
    'val_precision': eval_results['eval_precision'],
    'val_recall': eval_results['eval_recall'],
    'val_accuracy': eval_results['eval_accuracy'],
    'train_time_minutes': train_result.metrics['train_runtime'] / 60,
}

results_df = pd.DataFrame([results_summary])
results_file = f"{args.output_dir}/training_results.csv"

if os.path.exists(results_file):
    existing = pd.read_csv(results_file)
    results_df = pd.concat([existing, results_df], ignore_index=True)

results_df.to_csv(results_file, index=False)
print(f"✓ Results saved to: {results_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"\nRun: {args.run_name}")
print(f"Hyperparameters: LR={args.learning_rate}, Epochs={args.epochs}, Batch={args.batch_size}")
print(f"\nPerformance on GameTox validation set:")
print(f"  F1:        {eval_results['eval_f1']:.4f}")
print(f"  Precision: {eval_results['eval_precision']:.4f}")
print(f"  Recall:    {eval_results['eval_recall']:.4f}")

print(f"\nModel saved to: {final_model_path}")
print(f"Next step: Run evaluate_modernbert.py to test on external datasets")

print("\n" + "=" * 70)
