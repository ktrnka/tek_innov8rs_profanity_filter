"""Toxic-BERT model (pre-trained baseline)."""

import time
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import BaseModel
from ..detector import ToxicityResult


class ToxicBERTModel(BaseModel):
    """Toxic-BERT pre-trained model for toxicity detection.

    Best cross-domain generalization (F1=0.67 on external datasets).
    Already trained, no custom fine-tuning needed.
    """

    def __init__(self):
        super().__init__("toxic-bert")
        self._model = None
        self._tokenizer = None
        self._device = None
        self._load()

    def _load(self):
        """Load Toxic-BERT from HuggingFace Hub."""
        model_name = "unitary/toxic-bert"

        print(f"Loading Toxic-BERT from HuggingFace Hub...")

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        # Move to appropriate device
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

        print(f"✓ Loaded Toxic-BERT on {self._device}")

    def predict(self, text: str) -> ToxicityResult:
        """Predict toxicity using Toxic-BERT."""
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

            # Toxic-BERT is binary: 0=Clean, 1=Toxic
            pred_class = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence = probabilities[pred_class].item()

            is_toxic = pred_class == 1

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
        """Predict toxicity for batch using Toxic-BERT."""
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
                is_toxic = pred_class == 1

                results.append(ToxicityResult(
                    text=text,
                    is_toxic=bool(is_toxic),
                    confidence=float(confidence),
                    toxicity_type=None,
                    latency_ms=latency_per_msg,
                    model_name=self.model_name
                ))

        return results
