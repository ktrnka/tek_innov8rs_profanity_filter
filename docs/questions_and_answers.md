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

## Fine-Tuning Memory Issues

**Problem:** ModernBERT fine-tuning runs out of memory and causes system thrashing.

**Solutions:**
1. **Start with small data subset** - Test with partial dataset before running overnight jobs
2. **Reduce batch size** - Try batch_size=1 if needed just to get it working
3. **Use smaller model** - DistilBERT (67M params) vs ModernBERT (149M params)
   - [DistilBERT base uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
   - [Search for models <100M params](https://huggingface.co/models?pipeline_tag=text-classification&num_parameters=min:0,max:0.1B&sort=downloads)

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
