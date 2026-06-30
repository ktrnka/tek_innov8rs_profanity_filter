"""Model loader factory."""

from .base import BaseModel
from .traditional_ml import TraditionalMLModel
from .modernbert import ModernBERTModel


def load_model(model_name: str) -> BaseModel:
    """Load a profanity detection model by name.

    Args:
        model_name: Name of the model to load
            - 'traditional-ml': Traditional ML (Level 3, fastest)
            - 'modernbert-binary': ModernBERT Binary (Level 4)
            - 'modernbert-multiclass': ModernBERT Multi-Class (Level 4, best)
            - 'toxic-bert': Toxic-BERT (pre-trained, best generalization)

    Returns:
        Loaded BaseModel instance

    Raises:
        ValueError: If model_name is unknown
    """
    if model_name == "traditional-ml":
        return TraditionalMLModel()

    elif model_name == "modernbert-binary":
        return ModernBERTModel(variant="binary")

    elif model_name == "modernbert-multiclass":
        return ModernBERTModel(variant="multiclass")

    elif model_name == "toxic-bert":
        # Toxic-BERT uses same interface as ModernBERT binary
        # but loads from HuggingFace Hub
        from .toxic_bert import ToxicBERTModel
        return ToxicBERTModel()

    else:
        available = [
            'traditional-ml',
            'modernbert-binary',
            'modernbert-multiclass',
            'toxic-bert'
        ]
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(available)}"
        )
