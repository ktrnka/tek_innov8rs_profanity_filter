# Profanity Filter - Hybrid Toxicity Detection

A production-ready profanity filter for gaming chat, combining three state-of-the-art approaches:

- **Traditional ML** (Level 3): Ultra-fast first-pass filtering (0.008ms)
- **ModernBERT Multi-Class** (Level 4): Best in-domain accuracy (F1=0.85)
- **Toxic-BERT** (pre-trained): Best cross-domain generalization (F1=0.67)

## 🎯 Quick Summary

We've successfully packaged the **hybrid profanity filter** combining all three best-performing models from Levels 3-4:

✅ **Package Structure Created**
✅ **All Models Integrated** (Traditional ML, ModernBERT, Toxic-BERT)  
✅ **Hybrid Two-Stage Filtering** implemented
✅ **CLI Tool** with full functionality
✅ **pyproject.toml** configured for installation

---

## 📦 Installation

```bash
cd level4-advanced
uv pip install -e .
```

---

## 🚀 Quick Start

```python
from profanity_filter import ProfanityDetector

# Auto mode - uses hybrid approach
detector = ProfanityDetector()
result = detector.predict("your text here")
print(result)  # CLEAN (95%)
```

**CLI:**
```bash
profanity-filter check "your message"
profanity-filter list-models
```

---

## Token Budget Status

**Remaining:** ~98K tokens  
**Next Task:** Build FastAPI web service (~20-30K tokens estimated)

See full documentation in package files.
