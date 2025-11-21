#!/usr/bin/env python3
"""
Profanity Filter CLI

Main entry point for training, evaluating, and using profanity filters.
"""

import click
from data_loader import load_gametox, get_binary_labels, ToxicityLabel


@click.group()
def cli():
    """Profanity filter for text classification."""
    pass


@cli.command()
def stats():
    """Show statistics about the GameTox dataset.
    
    Usage: uv run main.py stats
    """
    try:
        df = load_gametox()
        click.echo(f"Total messages: {len(df):,}")
        click.echo("\nLabel distribution:")
        for label in ToxicityLabel:
            count = (df['label'] == label).sum()
            pct = 100 * count / len(df)
            click.echo(f"  {label.name}: {count:,} ({pct:.1f}%)")
        
        binary_labels = get_binary_labels(df)
        toxic_count = binary_labels.sum()
        click.echo("\nBinary classification:")
        click.echo(f"  Toxic: {toxic_count:,} ({100*toxic_count/len(df):.1f}%)")
        click.echo(f"  Non-toxic: {len(df)-toxic_count:,} ({100*(len(df)-toxic_count)/len(df):.1f}%)")
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


# ============================================================================
# Regex Filter Commands
# ============================================================================

@cli.group()
def regex():
    """Rule-based regex filter (Level 1)."""
    pass


@regex.command()
@click.argument("text")
def predict(text):
    """Test the regex filter on a single text.
    
    Usage: uv run main.py regex predict "this is a test"
    """
    click.echo(f"Processing: {text}")
    # TODO: Implement regex filtering logic
    click.echo("Result: (not implemented yet)")


@regex.command()
def evaluate():
    """Evaluate the regex filter on GameTox test data.
    
    Usage: uv run main.py regex evaluate
    """
    click.echo("Evaluating regex filter...")
    # TODO: Implement evaluation logic
    click.echo("(not implemented yet)")


@regex.command()
@click.argument("word")
def count_word(word):
    """Count messages containing a specific word.
    
    Usage: uv run main.py regex count-word damn
    """
    try:
        df = load_gametox()
        binary_labels = get_binary_labels(df)
        
        # Find messages containing the word (case-insensitive)
        contains_word = df['message'].str.contains(word, case=False, na=False)
        
        total_with_word = contains_word.sum()
        toxic_with_word = (contains_word & binary_labels).sum()
        non_toxic_with_word = (contains_word & ~binary_labels).sum()
        
        click.echo(f"Messages containing '{word}': {total_with_word:,}")
        click.echo(f"  Labeled as toxic: {toxic_with_word:,}")
        click.echo(f"  Labeled as non-toxic: {non_toxic_with_word:,}")
        
        if total_with_word > 0:
            click.echo(f"\nPercentage toxic: {100*toxic_with_word/total_with_word:.1f}%")
            
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


# ============================================================================
# Sklearn Filter Commands
# ============================================================================

@cli.group()
def sklearn():
    """Traditional ML classifier using scikit-learn (Level 3)."""
    pass


@sklearn.command()
@click.option("--output", default="models/sklearn_filter.pkl", help="Output path for trained model")
@click.option("--test-size", default=0.2, help="Fraction of data to use for testing")
def train(output, test_size):
    """Train a scikit-learn text classifier.
    
    Usage: uv run main.py sklearn train --output models/my_model.pkl
    """
    click.echo(f"Training sklearn model...")
    click.echo(f"Test size: {test_size}")
    click.echo(f"Will save to: {output}")
    # TODO: Implement training logic


@sklearn.command()
@click.argument("text")
@click.option("--model", default="models/sklearn_filter.pkl", help="Path to trained model")
def predict(text, model):
    """Test the sklearn filter on a single text.
    
    Usage: uv run main.py sklearn predict "this is a test"
    """
    click.echo(f"Processing: {text}")
    click.echo(f"Using model: {model}")
    # TODO: Implement prediction logic
    click.echo("Result: (not implemented yet)")


@sklearn.command()
@click.option("--model", default="models/sklearn_filter.pkl", help="Path to trained model")
def evaluate(model):
    """Evaluate the sklearn filter on test data.
    
    Usage: uv run main.py sklearn evaluate
    """
    click.echo(f"Evaluating sklearn model: {model}")
    # TODO: Implement evaluation logic
    click.echo("(not implemented yet)")


if __name__ == "__main__":
    cli()

