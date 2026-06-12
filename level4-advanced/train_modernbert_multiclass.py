#!/usr/bin/env python3
"""
Fine-tune ModernBERT for multi-class toxicity classification.

Classes:
- 0: Clean/Normal
- 1: Profanity
- 2: Insult
- 3: Hate Speech (includes rare classes 4, 5)

This allows distinguishing between different types of toxicity.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
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
print("MODERNBERT MULTI-CLASS TOXICITY CLASSIFICATION")
print("=" * 70)

# ============================================================================
# Argument Parsing
# ============================================================================

parser = argparse.ArgumentParser(description='Fine-tune ModernBERT for multi-class')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate (default: 2e-5)')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs (default: 3)')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Training batch size (default: 4)')
parser.add_argument('--max_length', type=int, default=512,
                    help='Maximum sequence length (default: 512)')
parser.add_argument('--output_dir', type=str, default='./modernbert_multiclass',
                    help='Output directory for model')
parser.add_argument('--run_name', type=str, default='run1_4class',
                    help='Run name for tracking')

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

gametox_path = '../data/GameTox/gametox.csv'
df = pd.read_csv(gametox_path)

print(f"\nLoaded {len(df)} messages from GameTox")

# Remove any NaN values FIRST
df = df.dropna()

# Merge rare classes 4, 5 into class 3 (hate speech)
df['label'] = df['label'].astype(int)
df.loc[df['label'] > 3, 'label'] = 3

# Rename message to text for consistency
df = df[['message', 'label']].rename(columns={'message': 'text'})

print(f"\nMulti-class label distribution:")
for label in sorted(df['label'].unique()):
    count = (df['label'] == label).sum()
    pct = count / len(df) * 100
    label_name = {0: 'Clean', 1: 'Profanity', 2: 'Insult', 3: 'Hate Speech'}[label]
    print(f"  {label}: {label_name:12s} {count:6d} ({pct:5.1f}%)")

# Train/validation split (80/20, stratified)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nTrain set: {len(train_df)} messages")
print(f"Validation set: {len(val_df)} messages")

# ============================================================================
# Load Model and Tokenizer
# ============================================================================

print("\n" + "=" * 70)
print("LOADING MODERNBERT MODEL")
print("=" * 70)

model_name = "answerdotai/ModernBERT-base"

print(f"\nLoading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model for 4-class classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,  # 4 classes: Clean, Profanity, Insult, Hate Speech
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
    """Compute multi-class metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Macro averaging (treats all classes equally)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

    # Weighted averaging (accounts for class imbalance)
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)

    accuracy = accuracy_score(labels, predictions)

    return {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
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
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=2,
    fp16=False,
    dataloader_num_workers=0,
    remove_unused_columns=True,
)

print(f"\nTraining arguments:")
print(f"  Output dir: {output_dir}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Epochs: {training_args.num_train_epochs}")

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
print("\n⚠️  This will take ~2-3 hours. Training runs in background.")
print("    You can monitor progress in the terminal.\n")

# Train!
train_result = trainer.train()

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\nTraining metrics:")
print(f"  Total time: {train_result.metrics['train_runtime']:.1f}s ({train_result.metrics['train_runtime'] / 60:.1f} min)")
print(f"  Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

# ============================================================================
# Evaluate on Validation Set
# ============================================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

eval_results = trainer.evaluate()

print(f"\nValidation metrics:")
print(f"  Macro F1:      {eval_results['eval_f1_macro']:.4f}")
print(f"  Weighted F1:   {eval_results['eval_f1_weighted']:.4f}")
print(f"  Accuracy:      {eval_results['eval_accuracy']:.4f}")

# Get detailed per-class metrics
val_texts = val_df['text'].tolist()
val_labels = val_df['label'].values

print("\nGenerating per-class report...")
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

class_names = ['Clean', 'Profanity', 'Insult', 'Hate Speech']
report = classification_report(val_labels, pred_labels, target_names=class_names, zero_division=0)
print("\n" + report)

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
    'val_f1_macro': eval_results['eval_f1_macro'],
    'val_f1_weighted': eval_results['eval_f1_weighted'],
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
print(f"  Macro F1:      {eval_results['eval_f1_macro']:.4f}")
print(f"  Weighted F1:   {eval_results['eval_f1_weighted']:.4f}")
print(f"  Accuracy:      {eval_results['eval_accuracy']:.4f}")

print(f"\nModel saved to: {final_model_path}")
print(f"Next step: Evaluate on external datasets and compare with binary model")

print("\n" + "=" * 70)
