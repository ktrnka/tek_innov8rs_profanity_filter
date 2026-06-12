"""ModernBERT models (Level 4)."""

import time
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import BaseModel
from ..detector import ToxicityResult


class ModernBERTModel(BaseModel):
    """ModernBERT transformer models (binary or multi-class).

    Two variants:
    - Binary: Clean vs Toxic (F1=0.78)
    - Multi-Class: Clean, Profanity, Insult, Hate Speech (F1=0.85)
    """

    # Class names for multi-class model
    CLASS_NAMES = ["Clean", "Profanity", "Insult", "Hate Speech"]

    def __init__(self, variant: str = "multiclass"):
        """Initialize ModernBERT model.

        Args:
            variant: 'binary' or 'multiclass'
        """
        if variant not in ["binary", "multiclass"]:
            raise ValueError(f"variant must be 'binary' or 'multiclass', got {variant}")

        super().__init__(f"modernbert-{variant}")
        self.variant = variant
        self.is_multiclass = (variant == "multiclass")

        self._model = None
        self._tokenizer = None
        self._device = None
        self._load()

    def _load(self):
        """Load the trained ModernBERT model."""
        # Find model path
        package_root = Path(__file__).parent.parent.parent.parent

        if self.variant == "binary":
            model_path = package_root / "modernbert_finetuned" / "final_batch4" / "final_model"
        else:  # multiclass
            model_path = package_root / "modernbert_multiclass" / "run1_4class" / "final_model"

        if not model_path.exists():
            raise FileNotFoundError(
                f"ModernBERT {self.variant} model not found at {model_path}. "
                f"Please train it first using train_modernbert{'_multiclass' if self.is_multiclass else ''}.py"
            )

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self._model.eval()

        # Move to appropriate device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

        print(f"✓ Loaded ModernBERT {self.variant} on {self._device}")

    def predict(self, text: str) -> ToxicityResult:
        """Predict toxicity using ModernBERT."""
        start_time = time.time()

        # Tokenize
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self._device)

        # Predict
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

            # Get prediction and confidence
            pred_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence = probabilities[pred_class].item()

            if self.is_multiclass:
                # Multi-class: 0=Clean, 1=Profanity, 2=Insult, 3=Hate Speech
                is_toxic = pred_class > 0
                toxicity_type = self.CLASS_NAMES[pred_class]
            else:
                # Binary: 0=Clean, 1=Toxic
                is_toxic = pred_class == 1
                toxicity_type = None

        latency_ms = (time.time() - start_time) * 1000

        return ToxicityResult(
            text=text,
            is_toxic=is_toxic,
            confidence=confidence,
            toxicity_type=toxicity_type,
            latency_ms=latency_ms,
            model_name=self.model_name
        )

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[ToxicityResult]:
        """Predict toxicity for batch using ModernBERT."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            start_time = time.time()

            # Tokenize batch
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self._device)

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Get predictions and confidences
                pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            batch_time = time.time() - start_time
            latency_per_msg = (batch_time / len(batch_texts)) * 1000

            # Create results
            for text, pred_class, probs in zip(batch_texts, pred_classes, probabilities):
                confidence = probs[pred_class]

                if self.is_multiclass:
                    is_toxic = pred_class > 0
                    toxicity_type = self.CLASS_NAMES[pred_class]
                else:
                    is_toxic = pred_class == 1
                    toxicity_type = None

                results.append(ToxicityResult(
                    text=text,
                    is_toxic=bool(is_toxic),
                    confidence=float(confidence),
                    toxicity_type=toxicity_type,
                    latency_ms=latency_per_msg,
                    model_name=self.model_name
                ))

        return results
