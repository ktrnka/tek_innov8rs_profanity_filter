"""Core profanity detection classes with hybrid support."""

from dataclasses import dataclass
from typing import Optional, Literal
import time


@dataclass
class ToxicityResult:
    """Result of toxicity detection.

    Attributes:
        text: The input text that was analyzed
        is_toxic: Whether the text is toxic (binary)
        confidence: Confidence score (0-1)
        toxicity_type: Type of toxicity for multi-class models
                      (Clean, Profanity, Insult, Hate Speech)
        latency_ms: Time taken for prediction in milliseconds
        model_name: Name of the model used
        two_stage: Whether hybrid two-stage filtering was used
    """
    text: str
    is_toxic: bool
    confidence: float
    toxicity_type: Optional[str] = None
    latency_ms: Optional[float] = None
    model_name: Optional[str] = None
    two_stage: bool = False

    def __str__(self):
        status = 'TOXIC' if self.is_toxic else 'CLEAN'
        if self.toxicity_type and self.toxicity_type != 'Clean':
            return f"{status} ({self.toxicity_type}, {self.confidence:.2%})"
        return f"{status} ({self.confidence:.2%})"


class ProfanityDetector:
    """Hybrid profanity detection supporting multiple models.

    Supports three detection approaches:
    1. Traditional ML (fast, Level 3) - 0.008ms latency
    2. ModernBERT Multi-Class (accurate, Level 4) - F1=0.85
    3. Toxic-BERT (generalizes, pre-trained) - F1=0.67

    Can operate in single-model or hybrid two-stage mode.
    """

    AVAILABLE_MODELS = {
        'traditional-ml': 'Traditional ML (Level 3)',
        'modernbert-binary': 'ModernBERT Binary (Level 4)',
        'modernbert-multiclass': 'ModernBERT Multi-Class (Level 4)',
        'toxic-bert': 'Toxic-BERT (pre-trained)',
    }

    def __init__(
        self,
        model: Optional[str] = None,
        mode: Literal["single", "hybrid", "auto"] = "auto",
        fast_model: str = "traditional-ml",
        accurate_model: str = "modernbert-multiclass",
        confidence_threshold: float = 0.7
    ):
        """Initialize detector.

        Args:
            model: Specific model to use (overrides mode)
            mode: Detection mode
                - 'single': Use one model only
                - 'hybrid': Two-stage filtering (fast then accurate)
                - 'auto': Choose best approach based on context
            fast_model: Model for first-pass filtering in hybrid mode
            accurate_model: Model for second-pass in hybrid mode
            confidence_threshold: Threshold for hybrid mode (0-1)
        """
        self.mode = mode
        self.fast_model_name = fast_model
        self.accurate_model_name = accurate_model
        self.confidence_threshold = confidence_threshold

        # Override mode if specific model requested
        if model:
            self.mode = "single"
            self.model_name = model
        elif mode == "auto":
            # Auto mode uses hybrid for best performance
            self.mode = "hybrid"
            self.model_name = None
        else:
            self.model_name = model

        # Lazy loading - models loaded on first use
        self._models = {}

    def _load_model(self, model_name: str):
        """Load a model (cached)."""
        if model_name in self._models:
            return self._models[model_name]

        from .models import load_model
        print(f"Loading {model_name}...")
        model = load_model(model_name)
        self._models[model_name] = model
        return model

    def predict(self, text: str) -> ToxicityResult:
        """Predict toxicity for a single text.

        Args:
            text: The text to analyze

        Returns:
            ToxicityResult with prediction and metadata
        """
        start_time = time.time()

        if self.mode == "single":
            # Single model mode
            model = self._load_model(self.model_name)
            result = model.predict(text)

        elif self.mode == "hybrid":
            # Two-stage filtering
            # Stage 1: Fast model for initial screening
            fast_model = self._load_model(self.fast_model_name)
            fast_result = fast_model.predict(text)

            # If clean with high confidence, skip stage 2
            if not fast_result.is_toxic and fast_result.confidence >= self.confidence_threshold:
                result = fast_result
                result.two_stage = True
                result.model_name = f"{self.fast_model_name} (fast-pass)"
            else:
                # Stage 2: Accurate model for suspicious messages
                accurate_model = self._load_model(self.accurate_model_name)
                result = accurate_model.predict(text)
                result.two_stage = True
                result.model_name = f"{self.accurate_model_name} (verified)"

        # Update latency to include any overhead
        result.latency_ms = (time.time() - start_time) * 1000

        return result

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> list[ToxicityResult]:
        """Predict toxicity for multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of ToxicityResult objects
        """
        results = []

        if self.mode == "hybrid":
            # Hybrid batch processing
            fast_model = self._load_model(self.fast_model_name)

            # Stage 1: Screen all with fast model
            fast_results = fast_model.predict_batch(texts, batch_size)

            # Identify messages needing stage 2
            needs_verification = []
            verified_results = []

            for i, (text, fast_result) in enumerate(zip(texts, fast_results)):
                if not fast_result.is_toxic and fast_result.confidence >= self.confidence_threshold:
                    # Clean with high confidence - accept fast result
                    fast_result.two_stage = True
                    fast_result.model_name = f"{self.fast_model_name} (fast-pass)"
                    verified_results.append((i, fast_result))
                else:
                    # Needs verification
                    needs_verification.append((i, text))

            # Stage 2: Verify suspicious messages with accurate model
            if needs_verification:
                accurate_model = self._load_model(self.accurate_model_name)
                verify_texts = [text for _, text in needs_verification]
                verify_results = accurate_model.predict_batch(verify_texts, batch_size)

                for (i, _), result in zip(needs_verification, verify_results):
                    result.two_stage = True
                    result.model_name = f"{self.accurate_model_name} (verified)"
                    verified_results.append((i, result))

            # Sort by original index
            verified_results.sort(key=lambda x: x[0])
            results = [result for _, result in verified_results]

        else:
            # Single model batch processing
            model = self._load_model(self.model_name)
            results = model.predict_batch(texts, batch_size)

        return results

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """List all available models.

        Returns:
            Dictionary mapping model names to descriptions
        """
        return cls.AVAILABLE_MODELS.copy()
