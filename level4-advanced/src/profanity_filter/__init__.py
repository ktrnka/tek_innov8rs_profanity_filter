"""
Profanity Filter - Hybrid Gaming Chat Toxicity Detection

A production-ready profanity filter combining multiple detection approaches:
- Traditional ML: Fast first-pass filtering (0.008ms, Level 3)
- ModernBERT Multi-Class: Best in-domain accuracy (F1=0.85, Level 4)
- Toxic-BERT: Best cross-domain generalization (F1=0.67, pre-trained)

Usage:
    # Simple auto-mode (uses hybrid approach)
    from profanity_filter import ProfanityDetector
    detector = ProfanityDetector()
    result = detector.predict("your text here")

    # Choose specific model
    detector = ProfanityDetector(model="modernbert-multiclass")

    # Hybrid two-stage filtering
    detector = ProfanityDetector(
        mode="hybrid",
        fast_model="traditional-ml",
        accurate_model="modernbert-multiclass"
    )
"""

__version__ = "0.1.0"

from .detector import ProfanityDetector, ToxicityResult

__all__ = [
    "ProfanityDetector",
    "ToxicityResult",
]
