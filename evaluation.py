"""
Shared evaluation utilities for profanity filters.
"""

import click
import pandas as pd
from sklearn.metrics import classification_report


def print_evaluation_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    messages: pd.Series,
    num_samples: int = 10,
    filter_name: str = "Filter"
):
    """Print a comprehensive evaluation report with classification metrics and error samples.
    
    Args:
        y_true: True binary labels (True = toxic, False = clean)
        y_pred: Predicted binary labels
        messages: The text messages corresponding to the predictions
        num_samples: Number of false positive/negative samples to display
        filter_name: Name of the filter for display purposes
    """
    # Print classification report
    click.echo("\n" + "="*60)
    click.echo("CLASSIFICATION REPORT")
    click.echo("="*60)
    report = classification_report(
        y_true, 
        y_pred,
        target_names=['Clean', 'Profane'],
        digits=3
    )
    click.echo(report)
    
    # Create DataFrame for easier filtering
    df = pd.DataFrame({
        'message': messages,
        'true': y_true,
        'pred': y_pred
    })
    
    # Find false positives and false negatives
    false_positives = df[df['pred'] & ~df['true']]
    false_negatives = df[~df['pred'] & df['true']]
    
    # Sample and display false positives
    click.echo("\n" + "="*60)
    click.echo("FALSE POSITIVES (flagged as profane, but actually clean)")
    click.echo(f"Showing {min(num_samples, len(false_positives))} of {len(false_positives)} total")
    click.echo("="*60)
    if len(false_positives) > 0:
        sample_fp = false_positives.sample(n=min(num_samples, len(false_positives)), random_state=42)
        for i, row in enumerate(sample_fp.itertuples(), 1):
            click.echo(f"{i:2d}. {row.message}")
    else:
        click.echo("None found!")
    
    # Sample and display false negatives
    click.echo("\n" + "="*60)
    click.echo("FALSE NEGATIVES (missed toxic messages)")
    click.echo(f"Showing {min(num_samples, len(false_negatives))} of {len(false_negatives)} total")
    click.echo("="*60)
    if len(false_negatives) > 0:
        sample_fn = false_negatives.sample(n=min(num_samples, len(false_negatives)), random_state=42)
        for i, row in enumerate(sample_fn.itertuples(), 1):
            click.echo(f"{i:2d}. {row.message}")
    else:
        click.echo("None found!")
    
    click.echo()
