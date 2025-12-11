# Questions and Answers

This document contains Q&A discussions from Discord, reformatted for reference. Student names have been removed for privacy.

---

## Working with LLM Rate Limits (OpenRouter)

**Question:**
I'm having issues with the free models in OpenRouter related to rate limits (requests/min and requests/day). I already hit those and the LLM profanity filter hung because the LLM started spitting out 429 rate limit exceeded errors. I'm using a very small sample from Gametox of only 100 messages vs 53K messages (entire dataset) and EVEN THEN I'm hitting the daily quota limit. Then it says to wait for 8000 seconds or 5000 seconds to retry.

- Velocity limit: 20 requests per minute
- Daily quota: 50 requests per day for free models

Do we need to just use a 50 messages sample so we are right at the daily limit? Or maybe send the 100 messages in bulk so it counts as 1 request? Would 100 messages be OK or do we need to send big numbers for relevant results and metrics?

**Keith's Response:**
Oh I didn't realize their rate limit was so low! Yeah I'd suggest just running on a sample in that case.

**Follow-up:**
Yes I'm going to batch it. 100 messages in 1 or 2 requests so I can scale up to 5000 messages in 1 day (in theory) (50 requests × 100 messages).

**Keith's Response:**
If they've got a batch API that's a great way to go. If you aren't changing the prompt too much you can save previous results to disk and only run requests for new ones. For small projects I often use https://grantjenks.com/docs/diskcache/tutorial.html for that.

If you're able to scale up to 5000 messages that should be plenty to compare against the regex-based filter. One thing you can do is to run the regex filter on the ones that you've already run through the LLM filter. That way you're comparing the two on exactly the same data.

**Follow-up:**
I did that on the entire dataset on level 1 so I would have to repeat all that with the same 5000 (if I even get there) of whatever amount for that matter.

Their batch API is for embeddings only (generating vector representations of text). We need chat completions (getting the LLM to classify text as toxic/clean). No batch API exists for chat completions.

**Keith's Response:**
Ah yeah I see. If I had to guess, the LLM will likely be quite a bit better than regex so even with say 100 examples you might see a numeric difference. Another option would be to pick 100 hard examples, which will make it easier to see the difference between different approaches.

Thinking about this some more, I'd say the most useful things to learn are calling LLM APIs (which you've already done) and get a sense of comparison between a LLM and regex approach. Beyond that, trying to work around rate limits isn't as important to learn.

I'll update the readme to summarize for anyone that hasn't gotten to level 2 yet.

---

## Dealing with Memory Issues When Fine-Tuning

**Question:**
Good evening. I've been trying to fine tune ModernBERT with batch size of 16 and 8. In both occasions it blew up my memory and my system got stuck on memory thrashing, basically failing (3+ hours lost). I'm doing a last attempt with batch_size 4 but I have to leave it running overnight for it to finish. If it fails again I guess I'm going the route of "cease and desist".

**Keith's Response:**
Things I'd try:

1. **Try fine-tuning on a small subset of the data** so that it doesn't take as long if it fails (only try training on everything once you feel good about it because yeah these things tend to run overnight)

2. **Decreasing batch size** like you're doing is good. Just to get it working you might try batch size 1

3. **Try a smaller model**, like https://huggingface.co/distilbert/distilbert-base-uncased is 67 million parameters instead of ModernBERT's 149 million parameters. You can even search for smaller ones to try. The Huggingface search UI doesn't give a slider below 1 billion params but you can edit the URL to go lower: https://huggingface.co/models?pipeline_tag=text-classification&num_parameters=min:0,max:0.1B&sort=downloads

---

## Finding Multilingual Profanity Datasets

**Question:**
I have a question regarding Level 4 language dataset. My thought is to have AI to go out and search for bad words in different languages and build a filter based on that. Is there a better approach? Is this the correct approach?

**Keith's Response:**
I'm not understanding the question entirely but I'll reply to what I can and maybe we can chat about it this evening or later. In industry it depends a lot on the amount of time you have, the existing code you have, and the business needs. So I'd say the correct approach depends greatly on the situation.

**Different approaches by filter type:**

- **LLM-based filter**: It can already filter profanity in other languages, though you may have to adjust your prompt a little bit to make it clear to filter non-English profanity, or provide guidance of what's profane in different languages or cultures.

- **Multilingual neural network (e.g., multilingual BERT)**: You can train it on English and because of the way embeddings work, it may work acceptably in many languages. In that case I'd focus just on testing it on different languages.

- **Word list or regex filter**: I usually start by searching for a profanity list in another language and then I use Google Translate, DeepL, or ChatGPT to double-check that it's a legit profanity list. Using ChatGPT to generate profanity might not work because they have safety features built-in to prevent usage for harassment and such. There are some ways around it, but you may not know if you can trust the list it produces without some other way to test it.

- **Scikit-learn approach with words**: If the dataset has some profanity in other languages, it'll learn those just fine.

**In all cases:**
It's helpful to run a language identifier on your data so that you can compute stats like "X% accurate in English, Y% accurate in French" so that you can identify which languages need the most work. Then for sklearn or BERT approaches I'd search the web for more data for the language that needs the most work.

**Part 2:**
If you're using a word list approach or regex approach, and find that it's less accurate in French, then you can review your existing word list for regular French words that are fine or profane French words to add.

Long story short, the approach depends on your existing path and the kinds of errors you're seeing. The important part is to find a systematic approach to identify & understand errors and improve your solution.

**Follow-up:**
Hi Keith, after doing some exploring I wound up using ChatGPT to list public locations I can gather datasets containing explicit foreign words. I had Claude code to do a comparison using the sklearn, toxic-BERT and ModernBERT models on the datasets.
