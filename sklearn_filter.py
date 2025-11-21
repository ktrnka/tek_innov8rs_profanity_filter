"""
Scikit-learn based profanity filter implementation (Level 3).
"""

from pathlib import Path
import pickle
import click
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from data_loader import load_gametox, get_binary_labels
from evaluation import print_evaluation_report


class SklearnProfanityFilter:
    """A machine learning-based profanity filter using scikit-learn."""
    
    def __init__(self, model_path: str | None = None):
        """Initialize the filter, optionally loading a trained model.
        
        Args:
            model_path: Path to a saved model pickle file
        """
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            # Create pipeline with specified hyperparameters
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 1),  # unigrams only
                    min_df=3,            # ignore terms appearing in < 3 documents
                    max_df=0.2,          # ignore terms appearing in > 20% of documents
                    lowercase=True,
                    strip_accents='unicode'
                )),
                ('classifier', LogisticRegressionCV(
                    cv=5,                # 5-fold cross-validation
                    random_state=42,
                    max_iter=1000,
                    n_jobs=-1            # use all CPU cores
                ))
            ])
    
    def train(self, X_train, y_train) -> 'SklearnProfanityFilter':
        """Train the filter on training data.
        
        Args:
            X_train: Training messages (list or Series of strings)
            y_train: Training labels (boolean Series)
            
        Returns:
            Self for method chaining
        """
        click.echo("Training TF-IDF vectorizer and logistic regression...")
        self.pipeline.fit(X_train, y_train)
        click.echo("Training complete!")
        return self
    
    def classify(self, text: str) -> bool:
        """Classify a single text as profane or not.
        
        Args:
            text: The text to classify
            
        Returns:
            True if profane, False if clean
        """
        return bool(self.pipeline.predict([text])[0])
    
    def predict(self, texts):
        """Classify multiple texts.
        
        Args:
            texts: List or Series of texts
            
        Returns:
            Array of boolean predictions
        """
        return self.pipeline.predict(texts)
    
    def save(self, filepath: str):
        """Save the trained model to a file.
        
        Args:
            filepath: Path where to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        click.echo(f"Model saved to: {filepath}")


@click.group()
def sklearn():
    """Traditional ML classifier using scikit-learn (Level 3)."""
    pass


@sklearn.command()
@click.option("--output", default="models/sklearn_filter.pkl", help="Output path for trained model")
@click.option("--test-size", default=0.2, help="Fraction of data to use for testing")
def train(output, test_size):
    """Train a scikit-learn text classifier.
    
    Usage: 
        uv run main.py sklearn train
        uv run main.py sklearn train --output models/my_model.pkl
    """
    try:
        click.echo("Loading GameTox dataset...")
        df = load_gametox()
        binary_labels = get_binary_labels(df)
        
        # Stratified train-test split
        click.echo(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test (stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'],
            binary_labels,
            test_size=test_size,
            random_state=42,
            stratify=binary_labels
        )
        
        click.echo(f"Training set: {len(X_train):,} messages ({y_train.sum():,} toxic)")
        click.echo(f"Test set: {len(X_test):,} messages ({y_test.sum():,} toxic)\n")
        
        # Train the model
        filter_obj = SklearnProfanityFilter()
        filter_obj.train(X_train, y_train)
        
        # Evaluate on test set
        click.echo("\nEvaluating on test set...")
        y_pred = filter_obj.predict(X_test)
        
        print_evaluation_report(
            y_test,
            y_pred,
            X_test,
            num_samples=10,
            filter_name="Sklearn Filter"
        )
        
        # Save the model
        filter_obj.save(output)
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@sklearn.command()
@click.argument("text")
@click.option("--model", default="models/sklearn_filter.pkl", help="Path to trained model")
def predict(text, model):
    """Test the sklearn filter on a single text.

    Usage: 
        uv run main.py sklearn predict "this is a test"
        uv run main.py sklearn predict "you noob" --model models/my_model.pkl
    """
    try:
        filter_obj = SklearnProfanityFilter(model_path=model)
        result = filter_obj.classify(text)
        
        click.echo(f"Text: {text}")
        click.echo(f"Result: {'PROFANE' if result else 'CLEAN'}")
        
    except FileNotFoundError:
        click.echo(f"Error: Model not found at {model}", err=True)
        click.echo("Train a model first with: uv run main.py sklearn train", err=True)
        raise click.Abort()


@sklearn.command()
@click.option("--model", default="models/sklearn_filter.pkl", help="Path to trained model")
@click.option("--samples", "-n", default=10, help="Number of false positive/negative samples to show")
def evaluate(model, samples):
    """Evaluate the sklearn filter on test data.

    Usage: 
        uv run main.py sklearn evaluate
        uv run main.py sklearn evaluate --model models/my_model.pkl --samples 20
    """
    try:
        click.echo("Loading GameTox dataset...")
        df = load_gametox()
        binary_labels = get_binary_labels(df)
        
        # Use same train-test split as training
        click.echo("Splitting data (same split as training)...")
        _, X_test, _, y_test = train_test_split(
            df['message'],
            binary_labels,
            test_size=0.2,
            random_state=42,
            stratify=binary_labels
        )
        
        click.echo(f"Loading model from: {model}")
        filter_obj = SklearnProfanityFilter(model_path=model)
        
        click.echo("Classifying test messages...")
        y_pred = filter_obj.predict(X_test)
        
        print_evaluation_report(
            y_test,
            y_pred,
            X_test,
            num_samples=samples,
            filter_name="Sklearn Filter"
        )
        
    except FileNotFoundError:
        click.echo(f"Error: Model not found at {model}", err=True)
        click.echo("Train a model first with: uv run main.py sklearn train", err=True)
        raise click.Abort()
