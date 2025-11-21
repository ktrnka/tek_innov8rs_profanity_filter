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
@click.argument("text")
def example(text):
    """Test the filter on a single text example.
    
    Usage: uv run main.py example "this is a test"
    """
    click.echo(f"Processing: {text}")
    # TODO: Implement filtering logic
    click.echo("Result: (not implemented yet)")


@cli.command()
@click.option("--model-type", default="regex", help="Type of model to train (regex, sklearn, llm)")
@click.option("--output", default="models/filter.pkl", help="Output path for trained model")
def train(model_type, output):
    """Train a profanity filter model.
    
    Usage: uv run main.py train --model-type sklearn
    """
    click.echo(f"Training {model_type} model...")
    click.echo(f"Will save to: {output}")
    # TODO: Implement training logic


@cli.command()
@click.option("--model", default="models/filter.pkl", help="Path to trained model")
@click.option("--data", default="data/test.csv", help="Test data path")
def evaluate(model, data):
    """Evaluate a trained model on test data.
    
    Usage: uv run main.py evaluate --model models/filter.pkl
    """
    click.echo(f"Evaluating model: {model}")
    click.echo(f"Using test data: {data}")
    # TODO: Implement evaluation logic


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


if __name__ == "__main__":
    cli()
