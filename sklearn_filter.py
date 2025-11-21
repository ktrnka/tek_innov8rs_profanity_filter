"""
Scikit-learn based profanity filter implementation (Level 3).
"""

from pathlib import Path
import pickle
import click
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from skl2onnx import to_onnx
import onnxruntime as rt

from data_loader import load_gametox, get_binary_labels, load_reddit_usernames
from evaluation import print_evaluation_report


class SklearnProfanityFilter:
    """A machine learning-based profanity filter using scikit-learn."""
    
    def __init__(self, model_path: str | None = None, use_char_ngrams: bool = False):
        """Initialize the filter, optionally loading a trained model.
        
        Args:
            model_path: Path to a saved model pickle file
            use_char_ngrams: If True, use character n-grams (1-4) instead of word unigrams
        """
        if model_path and Path(model_path).exists():
            with open(model_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            # Create vectorizer based on mode
            if use_char_ngrams:
                vectorizer = TfidfVectorizer(
                    analyzer='char',     # character-level n-grams
                    ngram_range=(1, 4),  # 1-4 character n-grams
                    min_df=3,            # ignore terms appearing in < 3 documents
                    max_df=0.2,          # ignore terms appearing in > 20% of documents
                    lowercase=True,
                    strip_accents=None   # ONNX requires None
                )
            else:
                vectorizer = TfidfVectorizer(
                    ngram_range=(1, 1),  # unigrams only
                    min_df=3,            # ignore terms appearing in < 3 documents
                    max_df=0.2,          # ignore terms appearing in > 20% of documents
                    lowercase=True,
                    strip_accents=None   # ONNX requires None
                )
            
            # Create pipeline with specified hyperparameters
            self.pipeline = Pipeline([
                ('tfidf', vectorizer),
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
    
    def save_onnx(self, filepath: str, sample_input: str = "sample text"):
        """Save the trained model in ONNX format.
        
        Args:
            filepath: Path where to save the ONNX model
            sample_input: Sample input text for type inference
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert to ONNX format
            # Use a single sample input for type inference
            onx = to_onnx(self.pipeline, np.array([sample_input]))
            
            with open(filepath, 'wb') as f:
                f.write(onx.SerializeToString())
            
            click.echo(f"ONNX model saved to: {filepath}")
        except Exception as e:
            click.echo(f"Warning: Failed to save ONNX model: {e}", err=True)
            click.echo("ONNX export may not be supported for this model configuration.", err=True)


@click.group()
def sklearn():
    """Traditional ML classifier using scikit-learn (Level 3)."""
    pass


@sklearn.command()
@click.option("--output", default="models/sklearn_filter.pkl", help="Output path for trained model")
@click.option("--test-size", default=0.2, help="Fraction of data to use for testing")
@click.option("--char-ngrams", is_flag=True, help="Use character n-grams (1-4) instead of word unigrams")
def train(output, test_size, char_ngrams):
    """Train a scikit-learn text classifier.
    
    Usage: 
        uv run main.py sklearn train
        uv run main.py sklearn train --output models/my_model.pkl
        uv run main.py sklearn train --char-ngrams --output models/sklearn_char_ngrams.pkl
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
        
        if char_ngrams:
            click.echo("Using character n-grams (1-4) instead of word unigrams")
        
        # Train the model
        filter_obj = SklearnProfanityFilter(use_char_ngrams=char_ngrams)
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
        
        # Also save in ONNX format
        onnx_output = output.replace('.pkl', '.onnx')
        # Use a sample from training data for type inference
        sample_text = X_train.iloc[0] if len(X_train) > 0 else "sample text"
        filter_obj.save_onnx(onnx_output, sample_input=sample_text)
        
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
@click.argument("text")
@click.option("--model", default="models/sklearn_filter.onnx", help="Path to ONNX model")
def predict_onnx(text, model):
    """Test prediction using ONNX model.

    Usage: 
        uv run main.py sklearn predict-onnx "this is a test"
        uv run main.py sklearn predict-onnx "you noob" --model models/sklearn_char_ngrams.onnx
    """
    try:
        if not Path(model).exists():
            click.echo(f"Error: ONNX model not found at {model}", err=True)
            click.echo("Train a model first with: uv run main.py sklearn train", err=True)
            raise click.Abort()
        
        click.echo(f"Loading ONNX model from: {model}")
        
        # Create ONNX Runtime session
        sess = rt.InferenceSession(model, providers=["CPUExecutionProvider"])
        
        # Get input and output names
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        click.echo(f"Input name: {input_name}")
        click.echo(f"Output name: {label_name}")
        
        # Make prediction
        # ONNX expects numpy array of strings
        pred_onx = sess.run([label_name], {input_name: np.array([text])})[0]
        
        result = bool(pred_onx[0])
        
        click.echo(f"\nText: {text}")
        click.echo(f"Result: {'PROFANE' if result else 'CLEAN'}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
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


@sklearn.command()
@click.option("--model", default="models/sklearn_filter.pkl", help="Path to trained model")
@click.option("--sample-size", default=10000, help="Number of usernames to evaluate (0 for all)")
@click.option("--review-count", "-n", default=50, help="Number of flagged usernames to review")
def eval_usernames(model, sample_size, review_count):
    """Evaluate the sklearn filter on Reddit usernames.
    
    Usage:
        uv run main.py sklearn eval-usernames
        uv run main.py sklearn eval-usernames --sample-size 100000 --review-count 50
        uv run main.py sklearn eval-usernames --sample-size 0  # Load all usernames
    
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
        
        click.echo(f"Loading model from: {model}")
        filter_obj = SklearnProfanityFilter(model_path=model)
        
        click.echo("Classifying usernames...")
        click.echo("Note: This model was trained on gaming chat messages, not usernames.\n")
        
        flagged = filter_obj.predict(usernames)
        
        flagged_count = flagged.sum()
        clean_count = len(flagged) - flagged_count
        
        click.echo("\n" + "=" * 60)
        click.echo("RESULTS")
        click.echo("=" * 60)
        click.echo(f"Total usernames evaluated: {len(usernames):,}")
        click.echo(f"Flagged as offensive: {flagged_count:,} ({flagged_count/len(usernames)*100:.2f}%)")
        click.echo(f"Clean: {clean_count:,} ({clean_count/len(usernames)*100:.2f}%)")
        
        # Show sample of flagged usernames
        if flagged_count > 0:
            click.echo("\n" + "=" * 60)
            click.echo("FLAGGED USERNAMES FOR MANUAL REVIEW")
            click.echo(f"Showing {min(review_count, flagged_count)} of {flagged_count} total")
            click.echo("=" * 60)
            click.echo("Review these to estimate precision (true positives / flagged):\n")
            
            flagged_usernames = usernames[flagged]
            sample = flagged_usernames.head(review_count)
            
            for i, username in enumerate(sample, 1):
                click.echo(f"{i:3}. {username}")
            
            click.echo("\n" + "=" * 60)
            click.echo("PRECISION ESTIMATION")
            click.echo("=" * 60)
            click.echo("Manually count how many of the above usernames are actually offensive.")
            click.echo(f"Precision ≈ (offensive count) / {min(review_count, flagged_count)}")
            click.echo("\nExpected offensive rate in general population: 0.1-5%")
            click.echo(f"If precision is high (>80%), estimated true offensive count: ~{int(flagged_count * 0.9):,}")
            click.echo(f"If precision is medium (50%), estimated true offensive count: ~{int(flagged_count * 0.5):,}")
        
    except FileNotFoundError:
        click.echo(f"Error: Model not found at {model}", err=True)
        click.echo("Train a model first with: uv run main.py sklearn train", err=True)
        raise click.Abort()
