"""
Regex-based profanity filter implementation (Level 1).
"""

import re
from pathlib import Path
import click

from data_loader import load_gametox, get_binary_labels, load_reddit_usernames
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
        
        # Build regex pattern with word boundaries for each word (for messages)
        # Use \b for word boundaries and re.IGNORECASE for case-insensitive matching
        escaped_words = [re.escape(word) for word in words]
        pattern = r'\b(?:' + '|'.join(escaped_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)
        
        # Build regex pattern without word boundaries (for usernames)
        # Usernames often concatenate words without spaces: "ihatethisgame"
        pattern_no_boundaries = r'(?:' + '|'.join(escaped_words) + r')'
        self.pattern_no_boundaries = re.compile(pattern_no_boundaries, re.IGNORECASE)
        
        # For capitalization-based matching, store the word list
        # We'll split usernames at case transitions and check each part
        self.words_lower = [word.lower() for word in words]
        
        self.word_count = len(words)
    
    def classify(self, text: str) -> bool:
        """Classify a text as profane or not.
        
        Args:
            text: The text to classify
            
        Returns:
            True if profane (contains profanity), False if clean
        """
        return bool(self.pattern.search(text))
    
    def classify_username(self, username: str) -> bool:
        """Classify a username as profane or not.
        
        Uses substring matching without word boundaries since usernames
        often concatenate words (e.g., "ihatethisgame").
        
        Args:
            username: The username to classify
            
        Returns:
            True if profane (contains profanity), False if clean
        """
        return bool(self.pattern_no_boundaries.search(username))
    
    def classify_username_caps(self, username: str) -> bool:
        """Classify a username as profane using capitalization as word boundaries.
        
        Treats case transitions (lowercase to uppercase) as word boundaries,
        which helps catch camelCase/PascalCase offensive words like
        "ChefBoyAreWeFucked" while avoiding false positives like "glassguru".
        
        Algorithm:
        1. Split username at capitalization boundaries and non-alphanumeric chars
           (e.g., "ChefBoyAreFucked" -> ["Chef", "Boy", "Are", "Fucked"])
        2. Check if any segment exactly matches a profane word (case-insensitive)
        
        Args:
            username: The username to classify
            
        Returns:
            True if profane (contains profanity at word/case boundaries), False if clean
        """
        # First split on non-alphanumeric characters (hyphens, underscores, etc)
        # Then split each part at capitalization boundaries
        # This handles both "password-is-weak" and "ChefBoyAreWeFucked"
        parts = re.split(r'[^a-zA-Z0-9]+', username)
        
        all_segments = []
        for part in parts:
            # Split at: uppercase after lowercase, uppercase before lowercase
            # This regex splits "ChefBoyAreWeFucked" into ["Chef", "Boy", "Are", "We", "Fucked"]
            segments = re.findall(r'[A-Z]*[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', part)
            all_segments.extend(segments)
        
        # Check if any segment exactly matches a profane word
        segments_lower = [seg.lower() for seg in all_segments if seg]
        return any(seg in self.words_lower for seg in segments_lower)
    
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


@regex.command()
@click.option("--wordlist", "-w", default=None, help="Path to profanity word list file")
@click.option("--sample-size", default=10000, help="Number of usernames to evaluate (0 for all)")
@click.option("--review-count", "-n", default=50, help="Number of flagged usernames to review")
@click.option("--method", type=click.Choice(['boundaries', 'substring', 'caps'], case_sensitive=False), 
              default='caps', help="Matching method: boundaries (\\b), substring (no boundaries), caps (case transitions)")
def eval_usernames(wordlist, sample_size, review_count, method):
    """Evaluate the regex filter on Reddit usernames.
    
    Usage:
        uv run main.py regex eval-usernames
        uv run main.py regex eval-usernames --method substring
        uv run main.py regex eval-usernames --wordlist data/en_profanity.txt --sample-size 50000 --method caps
        uv run main.py regex eval-usernames --sample-size 0  # Load all usernames
    
    Note: Usernames are not annotated, so we can only estimate precision by manual review.
    Expected offensive rate: 0.1-5% of usernames.
    """
    try:
        # Convert 0 or negative to None (load all)
        actual_sample_size = None if sample_size <= 0 else sample_size
        
        if actual_sample_size is None:
            click.echo("Loading ALL Reddit usernames (~25M)...")
            click.echo("This may take a few minutes...")
        else:
            click.echo(f"Loading {sample_size:,} Reddit usernames...")
        
        usernames = load_reddit_usernames(sample_size=actual_sample_size)
        
        if wordlist:
            filter_obj = RegexProfanityFilter(filepath=wordlist)
            click.echo(f"Using custom wordlist: {wordlist}")
        else:
            filter_obj = RegexProfanityFilter.basic_filter()
            click.echo("Using basic built-in wordlist")
        
        click.echo(f"Filter contains {filter_obj.word_count} profane words")
        
        # Select classification method
        if method == 'boundaries':
            click.echo("\nNote: Using standard word boundaries (\\b)")
            click.echo("      This requires hyphens, underscores, or other separators\n")
            flagged = usernames.apply(filter_obj.classify)
        elif method == 'substring':
            click.echo("\nNote: Using substring matching (no word boundaries)")
            click.echo("      This catches concatenated words like 'ihatethisgame'\n")
            flagged = usernames.apply(filter_obj.classify_username)
        else:  # caps
            click.echo("\nNote: Using capitalization transitions as word boundaries")
            click.echo("      This catches 'ChefBoyAreWeFucked' but not 'cannibalasfuck'\n")
            flagged = usernames.apply(filter_obj.classify_username_caps)
        
        click.echo("Classifying usernames...")
        
        flagged_count = flagged.sum()
        flagged_pct = 100 * flagged_count / len(usernames)
        
        click.echo("\n" + "="*60)
        click.echo("RESULTS")
        click.echo("="*60)
        click.echo(f"Total usernames evaluated: {len(usernames):,}")
        click.echo(f"Flagged as offensive: {flagged_count:,} ({flagged_pct:.2f}%)")
        click.echo(f"Clean: {len(usernames) - flagged_count:,} ({100-flagged_pct:.2f}%)")
        
        if flagged_count == 0:
            click.echo("\nNo offensive usernames found!")
            click.echo("This may indicate the filter is too strict with word boundaries")
            click.echo("or the sample doesn't contain offensive usernames.")
            return
        
        # Show flagged usernames for manual review
        flagged_usernames = usernames[flagged]
        review_sample = min(review_count, len(flagged_usernames))
        
        click.echo("\n" + "="*60)
        click.echo(f"FLAGGED USERNAMES FOR MANUAL REVIEW")
        click.echo(f"Showing {review_sample} of {len(flagged_usernames)} total")
        click.echo("="*60)
        click.echo("Review these to estimate precision (true positives / flagged):\n")
        
        sample = flagged_usernames.sample(n=review_sample, random_state=42)
        for i, username in enumerate(sample, 1):
            click.echo(f"{i:3d}. {username}")
        
        click.echo("\n" + "="*60)
        click.echo("PRECISION ESTIMATION")
        click.echo("="*60)
        click.echo("Manually count how many of the above usernames are actually offensive.")
        click.echo(f"Precision ≈ (offensive count) / {review_sample}")
        click.echo("\nExpected offensive rate in general population: 0.1-5%")
        click.echo(f"If precision is high (>80%), estimated true offensive count: ~{int(flagged_count * 0.9):,}")
        click.echo(f"If precision is medium (50%), estimated true offensive count: ~{int(flagged_count * 0.5):,}")
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
