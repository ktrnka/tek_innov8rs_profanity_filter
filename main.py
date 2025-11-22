#!/usr/bin/env python3
"""
Profanity Filter CLI

Main entry point for training, evaluating, and using profanity filters.
"""

import click
from data_loader import load_gametox, get_binary_labels, ToxicityLabel
from regex_filter import regex
from sklearn_filter import sklearn
from llm_filter import llm


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
cli.add_command(regex)

# ============================================================================
# Sklearn Filter Commands
# ============================================================================
cli.add_command(sklearn)

# ============================================================================
# LLM Filter Commands
# ============================================================================
cli.add_command(llm)


if __name__ == "__main__":
    cli()
