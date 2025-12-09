"""Base model interface."""

from abc import ABC, abstractmethod
from typing import List
from ..detector import ToxicityResult


class BaseModel(ABC):
    """Base class for all toxicity detection models."""

    def __init__(self, model_name: str):
        """Initialize model.

        Args:
            model_name: Name/identifier for this model
        """
        self.model_name = model_name

    @abstractmethod
    def predict(self, text: str) -> ToxicityResult:
        """Predict toxicity for a single text.

        Args:
            text: The text to analyze

        Returns:
            ToxicityResult with prediction
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[ToxicityResult]:
        """Predict toxicity for multiple texts.

        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing

        Returns:
            List of ToxicityResult objects
        """
        pass
