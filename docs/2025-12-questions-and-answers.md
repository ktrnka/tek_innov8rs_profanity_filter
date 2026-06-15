# Q&A Reference

---

## LLM Rate Limits (OpenRouter Free Tier)

**Problem:** Hitting OpenRouter rate limits (20 requests/min, 50 requests/day) with even small datasets.

**Solutions:**
- **Use small samples** - 100 examples is enough to see differences between LLM and regex approaches
- **Batch messages** - Send 100 messages per request to maximize daily quota (theoretically 5000 messages/day)
- **Cache results** - Use [diskcache](https://grantjenks.com/docs/diskcache/tutorial.html) to save API responses and avoid re-running
- **Pick hard examples** - 100 difficult cases will show differences better than random samples
- **Compare on same data** - Run regex filter on the same messages you used for LLM testing

**Note:** OpenRouter's batch API only supports embeddings, not chat completions. Learning to call LLM APIs and compare approaches is more valuable than working around rate limits.

---

## Fine-Tuning Memory & Speed Issues

**Problem:** Fine-tuning a transformer (e.g., ModernBERT) runs out of memory and/or trains very slowly — especially on older or low-RAM machines.

**Solutions (try in roughly this order):**
1. **Iterate on a small data subset first** - Get the whole training script working and tune its speed on a few hundred examples before launching a full (possibly overnight) run. Don't debug on the full dataset.
2. **Reduce the batch size** - Smaller batches use less memory; try `batch_size=8`, then `4`, then even `1` just to get it running. (Smaller batches can be slower per epoch, so balance against speed.)
3. **Shorten the input length** - Transformer cost grows with sequence length, and chat messages are short — capping `max_length` (e.g., 64 or 128 tokens, with truncation) cuts memory use and speeds training a lot.
4. **Use a smaller model** - A "tiny"/distilled model often gets most of the quality at a fraction of the cost:
   - [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) and other distilled models
   - [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased) (~67M params) vs ModernBERT (~149M params)
   - [Search for text-classification models <100M params](https://huggingface.co/models?pipeline_tag=text-classification&num_parameters=min:0,max:0.1B&sort=downloads)

---

## Multilingual Profanity Filtering

**Approach depends on your filter type:**

### LLM-based
- Already works in multiple languages
- Adjust prompt to clarify non-English profanity expectations

### Multilingual BERT
- Train on English; embeddings often transfer to other languages
- Focus on testing across languages

### Word List / Regex
- Search for existing profanity lists in target language
- Verify with Google Translate, DeepL, or ChatGPT
- Note: ChatGPT may refuse to generate profanity lists due to safety features

### Scikit-learn
- Automatically learns profanity in any language present in training data

### General Best Practices
- **Use language identification** to compute per-language accuracy metrics
- **Find weak languages** and gather more training data for them
- **Review and iterate** - Identify errors systematically and improve based on patterns
- **Validate word lists** by reviewing false positives/negatives in target languages
