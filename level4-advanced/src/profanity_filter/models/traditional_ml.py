"""Traditional ML model (Level 3)."""

import time
import pickle
from pathlib import Path
from typing import List
import numpy as np

from .base import BaseModel
from ..detector import ToxicityResult


class TraditionalMLModel(BaseModel):
    """Traditional ML classifier using TF-IDF + Logistic Regression.

    This is the fastest model (0.008ms) from Level 3, ideal for
    first-pass filtering in hybrid mode.
    """

    def __init__(self):
        super().__init__("traditional-ml")
        self._pipeline = None
        self._load()

    def _load(self):
        """Load the trained model from Level 3."""
        # Look for model in ../level3-traditional-ml/
        package_root = Path(__file__).parent.parent.parent.parent.parent
        model_path = package_root / "level3-traditional-ml" / "profanity_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Traditional ML model not found at {model_path}. "
                f"Please train it first by running Level 3 training script."
            )

        with open(model_path, 'rb') as f:
            self._pipeline = pickle.load(f)

        print(f"✓ Loaded Traditional ML model from {model_path}")

    def predict(self, text: str) -> ToxicityResult:
        """Predict toxicity using Traditional ML."""
        start_time = time.time()

        # Predict
        prediction = self._pipeline.predict([text])[0]
        probabilities = self._pipeline.predict_proba([text])[0]

        is_toxic = bool(prediction == 1)
        confidence = float(probabilities[1] if is_toxic else probabilities[0])

        latency_ms = (time.time() - start_time) * 1000

        return ToxicityResult(
            text=text,
            is_toxic=is_toxic,
            confidence=confidence,
            toxicity_type=None,  # Binary model
            latency_ms=latency_ms,
            model_name=self.model_name
        )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[ToxicityResult]:
        """Predict toxicity for batch (Traditional ML is fast enough for full batch)."""
        start_time = time.time()

        # Predict all at once
        predictions = self._pipeline.predict(texts)
        probabilities = self._pipeline.predict_proba(texts)

        total_time = time.time() - start_time
        latency_per_msg = (total_time / len(texts)) * 1000

        results = []
        for text, pred, probs in zip(texts, predictions, probabilities):
            is_toxic = bool(pred == 1)
            confidence = float(probs[1] if is_toxic else probs[0])

            results.append(ToxicityResult(
                text=text,
                is_toxic=is_toxic,
                confidence=confidence,
                toxicity_type=None,
                latency_ms=latency_per_msg,
                model_name=self.model_name
            ))

        return results
