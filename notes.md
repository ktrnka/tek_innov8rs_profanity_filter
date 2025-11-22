# Username Filtering Evaluation Results

## Comparison of Filtering Methods

Evaluated on 100,000 Reddit usernames (unlabeled data).

### REGEX METHODS

#### Method 1: Word Boundaries (`\b`)
- **Command**: `uv run main.py regex eval-usernames --method boundaries`
- **Flagged**: 0.02% (22 out of 100,000)
- **Precision**: ~85%
- **Pros**: High precision, fast
- **Cons**: Misses concatenated words like "ihatethisgame"
- **Use case**: When false positives are very costly

#### Method 2: Substring Matching (no boundaries)
- **Command**: `uv run main.py regex eval-usernames --method substring`
- **Flagged**: 3.51% (3,510 out of 100,000)
- **Precision**: ~45%
- **Pros**: Catches all occurrences, including concatenated words
- **Cons**: Many false positives ("glass", "password", "massage")
- **Use case**: When recall is critical and manual review is available

#### Method 3: Capitalization Boundaries ⭐ **RECOMMENDED**
- **Command**: `uv run main.py regex eval-usernames --method caps` (default)
- **Flagged**: 0.55% (554 out of 100,000)
- **Precision**: ~90-95%
- **Algorithm**: 
  - Splits usernames at capitalization transitions (lowercase → uppercase)
  - Also splits at non-alphanumeric characters (hyphens, underscores)
  - Checks if any segment exactly matches a profane word
- **Examples**:
  - ✅ Catches: `ChefBoyAreWeFucked`, `FuckYou`, `NoobPlayer`, `StupidWhiteMale`
  - ❌ Misses: `cannibalasfuck` (all lowercase, no boundaries)
  - ✅ Avoids: `glassguru7`, `password-is-weak`, `AssociateProf`
- **Pros**: Best precision/recall balance for regex approaches
- **Cons**: Misses all-lowercase concatenated words
- **Use case**: **Production username filtering** - best overall performance

### ML METHOD

#### Method 4: Sklearn TF-IDF + Logistic Regression (Word Unigrams)
- **Command**: `uv run main.py sklearn eval-usernames`
- **Model**: `models/sklearn_filter.pkl`
- **Flagged**: 0.07% (66 out of 100,000)
- **Precision**: ~70-80%
- **Model**: TfidfVectorizer (unigrams, min_df=3, max_df=0.2) + LogisticRegressionCV (5-fold CV)
- **GameTox Performance**: F1 0.696, Precision 82.4%, Recall 60.2%
- **Pros**: Very conservative, learned from data, good precision
- **Cons**: 
  - Trained on gaming chat messages, not usernames (domain mismatch)
  - Low recall on usernames
  - Word tokenization struggles with concatenated usernames
  - Hyphens treated as word separators, weakening signal
- **Examples**:
  - ✅ True positives: `Cock-enters-Pussy`, `pussy-fucker69`, `i-fuck-cats`, `Fat-Chicks-Are-Gross`
  - ❌ False positives: `trash-in-progress`, `katie-is-gay`, `Duck-Fat`, `ChowYun-Fat`
- **Use case**: Gaming chat messages, not ideal for usernames

#### Method 5: Sklearn TF-IDF + Logistic Regression (Character N-grams 1-4) ⭐ **BEST ML APPROACH**
- **Command**: `uv run main.py sklearn eval-usernames --model models/sklearn_char_ngrams.pkl`
- **Training**: `uv run main.py sklearn train --char-ngrams --output models/sklearn_char_ngrams.pkl`
- **Model**: `models/sklearn_char_ngrams.pkl`
- **Flagged**: 1.79% (1,790 out of 100,000)
- **Precision**: ~60-70% (estimated)
- **Model**: TfidfVectorizer (char n-grams 1-4, min_df=3, max_df=0.2) + LogisticRegressionCV (5-fold CV)
- **GameTox Performance**: F1 0.740, Precision 83.2%, Recall 66.6%
- **Pros**: 
  - Better performance on both GameTox and usernames
  - Handles concatenated words well (`fuckthatshit`)
  - Captures partial obfuscation (`fck`, `sht`)
  - Better generalization to unseen combinations
- **Cons**:
  - Lower precision than regex caps method on usernames
  - Character overlap can cause false positives
- **Examples**:
  - ✅ True positives: `fuckthatracistshit`, `FireyShitBucket`, `FuckYouGotMine`, `fuckthispussieshit`, `Imafaggotbutimnotgay`, `Shitservers`, `smarmyfuck`, `sexyfuckingthrowaway`, `Msbitch90`
  - ❌ False positives: `Fakehashish`, `DumberMonkey`, `WittyLoser`, `PerpetualNoob`
- **Use case**: Both gaming chat messages and usernames - best ML approach for both domains

## Key Observations

1. **Character n-gram model best for both domains**: 
   - GameTox: F1 0.740 (vs. 0.696 for word unigrams)
   - Usernames: 1.79% flagged with ~60-70% precision (vs. 0.07% for word unigrams)
2. **Regex caps method still best for high-precision username filtering** (0.55% flagged, 90-95% precision)
3. **Character n-grams solve concatenation problem**: Can detect profanity in `fuckthatshit`, `ihatethisgame`, etc.
4. **Domain mismatch matters less with char n-grams**: Character patterns transfer better than word patterns
5. **Precision/recall tradeoff is clear**: 
   - Substring matching: 3.51% flagged, 45% precision (high recall, low precision)
   - Char n-grams: 1.79% flagged, 60-70% precision (good balance for ML)
   - Caps boundaries: 0.55% flagged, 90-95% precision (best regex balance)
   - Word unigrams: 0.07% flagged, 70-80% precision (too conservative)

## Performance Summary Table

| Method | Flagged % | Precision | GameTox F1 | Best For |
|--------|-----------|-----------|------------|----------|
| Regex Word Boundaries | 0.02% | ~85% | 0.607 | Maximum precision |
| Regex Substring | 3.51% | ~45% | 0.607 | Maximum recall |
| **Regex Caps** ⭐ | **0.55%** | **90-95%** | **0.607** | **Username filtering** |
| Sklearn Word Unigrams | 0.07% | 70-80% | 0.696 | Chat messages only |
| **Sklearn Char N-grams** ⭐ | **1.79%** | **60-70%** | **0.740** | **Both domains (ML)** |

## Recommendations

- **For username filtering (high precision)**: Use regex caps method (`--method caps`)
- **For username filtering (ML approach)**: Use sklearn char n-grams model
- **For gaming chat messages**: Use sklearn char n-grams model (best F1 score)
- **For maximum recall**: Use substring matching with manual review
- **For maximum precision**: Use regex word boundaries

## Key Insights

### Why Character N-grams Perform Better

Character n-grams (1-4) capture patterns that word-based methods miss:

1. **Concatenated words**: `fuckthatshit` → contains n-grams like "fuck", "shit"
2. **Partial obfuscation**: `fck`, `sht` → similar character patterns to full words
3. **Better generalization**: Learns character-level toxicity patterns, not just whole words
4. **Domain transfer**: Character patterns transfer better from messages to usernames

However, character n-grams also increase false positives due to character overlap in legitimate words (e.g., "Fakehashish" contains character patterns similar to "shit").

### Regex vs. ML Trade-offs

**Regex Caps Method**:
- Pros: Interpretable, fast, high precision, no training data needed
- Cons: Requires manual wordlist curation, misses all-lowercase concatenations

**Sklearn Char N-grams**:
- Pros: Learns from data, handles obfuscation, better recall on concatenations
- Cons: Requires labeled training data, less interpretable, lower precision than regex caps

## Expected Offensive Rate

General population baseline: 0.1-5% of usernames are offensive
- Our caps method (0.55% flagged) is within this range
- Sklearn method (0.07%) is likely under-flagging
- Substring method (3.51%) is near the upper bound but with many false positives

# Serving from Javascript

- Failed: Running Transformers.js locally with npm/parcel. It looks like it may need some env variables but I'm not sure which. Without those, it generates invalid JS when packing.
- Worked: Loading Transformers.js from CDN.
- Worked: Running an existing BERT model.
- Failed: Running my model. It failed because it needed files like tokenizer.json, config.json. From some searches, it seems that Huggingface does not support tokenizers that are embedded within the ONNX file at all and there are no tools to convert