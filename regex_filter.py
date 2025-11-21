"""
Regex-based profanity filter implementation (Level 1).
"""

import re
from pathlib import Path
import click

from data_loader import load_gametox, get_binary_labels
from evaluation import print_evaluation_report


class RegexProfanityFilter:
    """A simple regex-based profanity filter using word boundaries."""
    
    def __init__(self, profanity_list: list[str] | None = None, filepath: str | None = None):
        """Initialize the filter with a list of profane words or from a file.
        
        Args:
            profanity_list: List of profane words to filter
            filepath: Path to a file with one profane word per line
            
        Raises:
            ValueError: If neither profanity_list nor filepath is provided
        """
        if profanity_list is None and filepath is None:
            raise ValueError("Must provide either profanity_list or filepath")
        
        if filepath is not None:
            # Load from file
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"Profanity list file not found: {filepath}")
            with open(path) as f:
                words = [line.strip() for line in f if line.strip()]
        else:
            words = profanity_list
        
        # Build regex pattern with word boundaries for each word
        # Use \b for word boundaries and re.IGNORECASE for case-insensitive matching
        escaped_words = [re.escape(word) for word in words]
        pattern = r'\b(?:' + '|'.join(escaped_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.word_count = len(words)
    
    def classify(self, text: str) -> bool:
        """Classify a text as profane or not.
        
        Args:
            text: The text to classify
            
        Returns:
            True if profane (contains profanity), False if clean
        """
        return bool(self.pattern.search(text))
    
    @classmethod
    def basic_filter(cls) -> 'RegexProfanityFilter':
        """Create a basic profanity filter with a predefined list of words.
        
        Returns:
            An instance of RegexProfanityFilter with basic profane words
        """
        basic_profanity = [
            "damn", "hell", "crap", "shit", "fuck",
            "ass", "bastard", "bitch", "dick"
        ]
        return cls(profanity_list=basic_profanity)


@click.group()
def regex():
    """Rule-based regex filter (Level 1)."""
    pass


@regex.command()
@click.argument("text")
@click.option("--wordlist", "-w", default=None, help="Path to profanity word list file")
def predict(text, wordlist):
    """Test the regex filter on a single text.

    Usage: 
        uv run main.py regex predict "this is a test"
        uv run main.py regex predict "this is a test" --wordlist data/en_profanity.txt
    """
    if wordlist:
        filter = RegexProfanityFilter(filepath=wordlist)
    else:
        filter = RegexProfanityFilter.basic_filter()
    
    result = filter.classify(text)
    
    click.echo(f"Text: {text}")
    click.echo(f"Result: {'PROFANE' if result else 'CLEAN'}")
    click.echo(f"(Using {filter.word_count} profane words)")


@regex.command()
@click.option("--wordlist", "-w", default=None, help="Path to profanity word list file")
@click.option("--samples", "-n", default=10, help="Number of false positive/negative samples to show")
def evaluate(wordlist, samples):
    """Evaluate the regex filter on GameTox test data.

    Usage: 
        uv run main.py regex evaluate
        uv run main.py regex evaluate --wordlist data/en_profanity.txt
        uv run main.py regex evaluate --wordlist data/en_profanity.txt --samples 50
    """
    try:
        click.echo("Loading GameTox dataset...")
        df = load_gametox()
        binary_labels = get_binary_labels(df)
        
        if wordlist:
            filter = RegexProfanityFilter(filepath=wordlist)
            click.echo(f"Using custom wordlist: {wordlist}")
        else:
            filter = RegexProfanityFilter.basic_filter()
            click.echo("Using basic built-in wordlist")
        
        click.echo(f"Filter contains {filter.word_count} profane words\n")
        
        click.echo("Classifying messages...")
        predictions = df['message'].apply(filter.classify)
        
        print_evaluation_report(
            binary_labels,
            predictions,
            df['message'],
            num_samples=samples,
            filter_name="Regex Filter"
        )
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
