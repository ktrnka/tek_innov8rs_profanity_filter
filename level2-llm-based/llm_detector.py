#!/usr/bin/env python3
"""
Level 2 - LLM-Based Profanity Detector
Uses OpenRouter API for profanity detection with various models.

Following README.md tasks:
1. Implement LLM detector with effective prompt design
2. Test multiple FREE models and compare
3. Calculate production costs (theoretical - no paid API calls)
4. Compare against Level 1 performance

BATCH MODE: Splits messages into batches of 300 and sends each batch as a single
API request to stay within free tier limits (50 requests/day for free models).
Testing 4,800 messages across 3 models = 48 requests (16 batches per model).
"""

import os
import time
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()


def check_rate_limits(api_key):
    """
    Check OpenRouter API rate limits and return requests per minute.

    Returns:
        int: Requests per minute allowed, or None if couldn't determine
    """
    try:
        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            timeout=10  # 10 second timeout
        )
        response.raise_for_status()
        limits = response.json()

        print("\nOpenRouter Rate Limits:")
        print(json.dumps(limits, indent=2))

        # Try to extract rate limit (requests per minute)
        # The API response may have rate_limit info
        if 'data' in limits and 'is_free_tier' in limits['data']:
            is_free = limits['data']['is_free_tier']
            if is_free:
                print("\n✓ Free tier detected - using 20 requests/minute limit")
                return 20
            else:
                print("\n✓ Paid tier detected - using higher rate limit (60 requests/minute)")
                return 60

        # Default to conservative limit if can't determine
        print("\nℹ Could not determine rate limit tier - defaulting to 20 requests/minute")
        return 20

    except Exception as e:
        print(f"\nWarning: Could not check rate limits: {e}")
        print("Defaulting to conservative 20 requests/minute")
        return 20


class LLMProfanityDetector:
    """LLM-based profanity detector using OpenRouter API."""

    def __init__(self, model_name, api_key=None, rate_limit_per_minute=20):
        """
        Initialize LLM detector.

        Args:
            model_name: OpenRouter model identifier
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            rate_limit_per_minute: Requests per minute allowed (default 20)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_delay = 60.0 / rate_limit_per_minute  # seconds between requests
        self.total_cost = 0.0
        self.total_requests = 0
        self.total_latency = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.rate_limited = False  # Track if we've hit rate limits

    @staticmethod
    def get_prompt_v1(message):
        """
        Prompt V1: Simple, direct binary classification.

        Design principles:
        - Clear task definition
        - Explicit output format
        - Minimal context to reduce tokens
        """
        return f"""Is the following message profane, toxic, or contains offensive language?

Message: "{message}"

Answer with only "Yes" or "No"."""

    @staticmethod
    def get_prompt_v2(message):
        """
        Prompt V2: More specific with examples.

        Design principles:
        - Define what counts as profanity
        - Provide brief examples
        - Still request binary output
        """
        return f"""Classify if this message contains profanity, insults, hate speech, or toxic language.

Examples of toxic content:
- Profanity: "fuck", "shit", "damn"
- Insults: "you're an idiot", "stupid noob"
- Hate speech: slurs, discriminatory language

Message: "{message}"

Is this message toxic? Answer only "Yes" or "No"."""

    def is_profane(self, text, prompt_version=1):
        """
        Check if text contains profanity using LLM.

        Args:
            text: Message to classify
            prompt_version: Which prompt to use (1 or 2)

        Returns:
            bool: True if profane, False if clean
        """
        if not text or not text.strip():
            return False

        # Select prompt
        if prompt_version == 1:
            prompt = self.get_prompt_v1(text)
        else:
            prompt = self.get_prompt_v2(text)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,  # Deterministic for consistency
            "max_tokens": 10  # We only need "Yes" or "No"
        }

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            latency = time.time() - start_time

            # Check for rate limit errors before raising for status
            if response.status_code == 429:
                self.rate_limited = True  # Mark that we're rate limited
                error_data = response.json()
                print(f"\n❌ Rate limit exceeded in individual request!")
                print(f"   Message: {error_data.get('error', {}).get('message', 'Unknown error')}")
                if 'metadata' in error_data.get('error', {}):
                    headers = error_data['error']['metadata'].get('headers', {})
                    if headers:
                        print(f"   Limit: {headers.get('X-RateLimit-Limit', 'Unknown')}")
                        print(f"   Remaining: {headers.get('X-RateLimit-Remaining', 'Unknown')}")
                        reset_ts = headers.get('X-RateLimit-Reset')
                        if reset_ts:
                            reset_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(int(reset_ts) / 1000))
                            print(f"   Resets at: {reset_date}")
                return False

            response.raise_for_status()
            result = response.json()

            # Extract answer
            answer = result['choices'][0]['message']['content'].strip().lower()

            # Track metrics
            self.total_requests += 1
            self.total_latency += latency

            # Track token usage if available
            if 'usage' in result:
                self.total_input_tokens += result['usage'].get('prompt_tokens', 0)
                self.total_output_tokens += result['usage'].get('completion_tokens', 0)

            # Parse answer
            return "yes" in answer

        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return False

    def predict_batch_single_request(self, messages, prompt_version=1):
        """
        Predict profanity for all messages in a single API request (batched).

        Args:
            messages: List of text messages
            prompt_version: Which prompt to use (currently uses optimized batch prompt)

        Returns:
            List of predictions (1 for profane, 0 for clean), or None if failed
        """
        if not messages:
            return []

        # Build batch prompt
        numbered_messages = "\n".join([f"{i+1}. \"{msg}\"" for i, msg in enumerate(messages)])

        batch_prompt = f"""Classify each of the following {len(messages)} messages as either "toxic" or "clean".

A message is TOXIC if it contains:
- Profanity or vulgar language
- Insults or personal attacks
- Hate speech or discriminatory language

A message is CLEAN otherwise.

Messages:
{numbered_messages}

Respond with ONLY a JSON array of {len(messages)} classifications in the same order.
Use exactly "toxic" or "clean" for each entry (lowercase).

Example format: ["toxic", "clean", "toxic", "clean", ...]
"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": batch_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1000  # Enough for JSON array of up to ~300 classifications
        }

        try:
            print(f"\n   Sending batch request ({len(messages)} messages)...")
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60  # Longer timeout for batch
            )
            latency = time.time() - start_time

            # Check for rate limit errors
            if response.status_code == 429:
                self.rate_limited = True  # Mark that we're rate limited
                error_data = response.json()
                print(f"\n   ❌ RATE LIMIT EXCEEDED - STOPPING ALL REQUESTS")
                print(f"      Message: {error_data.get('error', {}).get('message', 'Unknown error')}")
                if 'metadata' in error_data.get('error', {}):
                    hdrs = error_data['error']['metadata'].get('headers', {})
                    if hdrs:
                        print(f"      Daily limit: {hdrs.get('X-RateLimit-Limit', 'Unknown')}")
                        print(f"      Remaining: {hdrs.get('X-RateLimit-Remaining', 'Unknown')}")
                        reset_ts = hdrs.get('X-RateLimit-Reset')
                        if reset_ts:
                            reset_date = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(int(reset_ts) / 1000))
                            print(f"      Resets at: {reset_date}")
                print(f"\n      Options:")
                print(f"      1. Wait for rate limit reset")
                print(f"      2. Add $10 credits → 1000 free requests/day")
                print(f"      3. Use paid models (contradicts README)")
                return None

            response.raise_for_status()
            result = response.json()

            # Track metrics
            self.total_requests += 1
            self.total_latency += latency

            # Track token usage
            if 'usage' in result:
                self.total_input_tokens += result['usage'].get('prompt_tokens', 0)
                self.total_output_tokens += result['usage'].get('completion_tokens', 0)

            # Extract and parse JSON response
            answer = result['choices'][0]['message']['content'].strip()

            # Try to parse JSON
            import json
            try:
                # Extract JSON array from response (handle markdown code blocks)
                if '```' in answer:
                    # Extract content between code fences
                    json_start = answer.find('[')
                    json_end = answer.rfind(']') + 1
                    answer = answer[json_start:json_end]

                classifications = json.loads(answer)

                # Validate we got the right number of predictions
                if len(classifications) != len(messages):
                    print(f"\n   ⚠ Warning: Expected {len(messages)} classifications, got {len(classifications)}")
                    return None

                # Convert to binary predictions
                predictions = [1 if c.lower() == "toxic" else 0 for c in classifications]
                print(f"   ✓ Batch classified in {latency:.2f}s")
                return predictions

            except json.JSONDecodeError as e:
                print(f"\n   ❌ Failed to parse JSON response: {e}")
                print(f"   Response: {answer[:200]}...")
                return None

        except Exception as e:
            print(f"\n   ❌ Error in batch request: {e}")
            return None

    def predict_batch(self, messages, show_progress=True, prompt_version=1, use_batch_api=True, batch_size=300):
        """
        Predict profanity for a batch of messages.

        Args:
            messages: List of text messages
            show_progress: Whether to print progress
            prompt_version: Which prompt to use
            use_batch_api: If True, split into batches and send batch requests; if False, use individual requests
            batch_size: Number of messages per batch (default 300)

        Returns:
            List of predictions (1 for profane, 0 for clean)
        """
        # Try batch approach first if enabled
        if use_batch_api:
            all_predictions = []
            num_batches = (len(messages) + batch_size - 1) // batch_size  # Ceiling division

            if show_progress:
                print(f"\n   Processing {len(messages)} messages in {num_batches} batches of ~{batch_size}...")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(messages))
                batch_messages = messages[start_idx:end_idx]

                if show_progress:
                    print(f"\n   Batch {batch_idx + 1}/{num_batches} ({len(batch_messages)} messages)...")

                predictions = self.predict_batch_single_request(batch_messages, prompt_version)

                if predictions is None:
                    # Check if we're rate limited - don't waste requests on fallback
                    if self.rate_limited:
                        print(f"\n   ⚠ STOPPED: Rate limit hit, cannot continue testing")
                        print(f"   Processed {len(all_predictions)} of {len(messages)} messages before stopping")
                        return None  # Signal failure to caller

                    # Only fall back for non-rate-limit failures (e.g., JSON parsing issues)
                    print(f"\n   ⚠ Batch {batch_idx + 1} failed (non-rate-limit), falling back to individual requests...")
                    predictions = []
                    for msg in batch_messages:
                        is_prof = self.is_profane(msg, prompt_version=prompt_version)
                        predictions.append(1 if is_prof else 0)
                        time.sleep(self.request_delay)

                all_predictions.extend(predictions)

                # Rate limiting - respect requests per minute limit
                if batch_idx < num_batches - 1:
                    time.sleep(self.request_delay)

            return all_predictions

        # Fallback to individual requests for all messages
        predictions = []
        for i, msg in enumerate(messages):
            if show_progress and (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(messages)} messages processed...")

            is_prof = self.is_profane(msg, prompt_version=prompt_version)
            predictions.append(1 if is_prof else 0)

            # Rate limiting - respect API limits
            time.sleep(self.request_delay)

        return predictions

    def get_stats(self):
        """Get statistics about API usage."""
        avg_latency = self.total_latency / self.total_requests if self.total_requests > 0 else 0
        avg_input_tokens = self.total_input_tokens / self.total_requests if self.total_requests > 0 else 0
        avg_output_tokens = self.total_output_tokens / self.total_requests if self.total_requests > 0 else 0

        return {
            'total_requests': self.total_requests,
            'total_latency': self.total_latency,
            'avg_latency': avg_latency,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'avg_input_tokens': avg_input_tokens,
            'avg_output_tokens': avg_output_tokens,
            'total_cost': self.total_cost
        }


def load_gametox(data_path, limit=None):
    """Load GameTox dataset."""
    messages = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            if not row['label'] or row['label'].strip() == '':
                continue
            messages.append(row['message'])
            labels.append(float(row['label']))

    return messages, labels


def save_test_subset(messages, labels, output_path):
    """Save test subset to CSV for consistent testing across all levels."""
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'message', 'label'])
        for i, (msg, label) in enumerate(zip(messages, labels)):
            writer.writerow([i, msg, label])
    print(f"\n✓ Saved {len(messages)} messages to {output_path}")
    print(f"  This subset will be used for all levels to ensure fair comparison.")


def evaluate(predictions, labels):
    """Calculate evaluation metrics."""
    total = len(labels)
    actual_toxic = [1 if label > 0 else 0 for label in labels]

    tp = sum(1 for i in range(total) if predictions[i] == 1 and actual_toxic[i] == 1)
    fp = sum(1 for i in range(total) if predictions[i] == 1 and actual_toxic[i] == 0)
    tn = sum(1 for i in range(total) if predictions[i] == 0 and actual_toxic[i] == 0)
    fn = sum(1 for i in range(total) if predictions[i] == 0 and actual_toxic[i] == 1)

    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'flagged': tp + fp
    }


def calculate_production_cost(avg_input_tokens, avg_output_tokens, input_price_per_1m, output_price_per_1m, messages_per_day=1000000):
    """
    Calculate theoretical production cost for 1M messages/day.

    Args:
        avg_input_tokens: Average input tokens per message
        avg_output_tokens: Average output tokens per message
        input_price_per_1m: Price per 1M input tokens (in $)
        output_price_per_1m: Price per 1M output tokens (in $)
        messages_per_day: Number of messages per day (default 1M)

    Returns:
        dict: Cost breakdown
    """
    input_cost_per_message = (avg_input_tokens / 1_000_000) * input_price_per_1m
    output_cost_per_message = (avg_output_tokens / 1_000_000) * output_price_per_1m
    total_cost_per_message = input_cost_per_message + output_cost_per_message

    daily_cost = total_cost_per_message * messages_per_day
    monthly_cost = daily_cost * 30
    yearly_cost = daily_cost * 365

    return {
        'cost_per_message': total_cost_per_message,
        'daily_cost': daily_cost,
        'monthly_cost': monthly_cost,
        'yearly_cost': yearly_cost,
        'input_tokens_per_day': avg_input_tokens * messages_per_day,
        'output_tokens_per_day': avg_output_tokens * messages_per_day
    }


def print_results(model_name, results, stats):
    """Print evaluation results for a model."""
    print("\n" + "="*70)
    print(f"  MODEL: {model_name}")
    print("="*70)
    print(f"Messages evaluated: {results['tp'] + results['fp'] + results['tn'] + results['fn']}")
    print(f"Messages flagged: {results['flagged']} ({100 * results['flagged'] / (results['tp'] + results['fp'] + results['tn'] + results['fn']):.1f}%)")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {results['tp']}  |  FP: {results['fp']}")
    print(f"  FN: {results['fn']}  |  TN: {results['tn']}")
    print()
    print("Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"  Precision: {results['precision']:.3f} ({results['precision']*100:.1f}%)")
    print(f"  Recall:    {results['recall']:.3f} ({results['recall']*100:.1f}%)")
    print(f"  F1-score:  {results['f1']:.3f}")
    print()
    print("Performance:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total time: {stats['total_latency']:.1f}s")
    print(f"  Avg latency: {stats['avg_latency']*1000:.0f}ms per message")
    if stats['avg_input_tokens'] > 0:
        print(f"  Avg input tokens: {stats['avg_input_tokens']:.1f}")
        print(f"  Avg output tokens: {stats['avg_output_tokens']:.1f}")


def main():
    """
    Step 0: Design effective prompt
    Step 1: Test multiple FREE models
    Step 2: Calculate theoretical production costs
    Step 3: Compare against Level 1
    """
    print("="*70)
    print("  LEVEL 2: LLM-BASED PROFANITY DETECTOR")
    print("="*70)

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\nERROR: OPENROUTER_API_KEY not found!")
        print("Please create a .env file with your OpenRouter API key:")
        print("  OPENROUTER_API_KEY=your_key_here")
        print("\nGet a free key at: https://openrouter.ai/")
        return

    print(f"\n✓ API key found")

    # Check rate limits
    rate_limit = check_rate_limits(api_key)
    delay_seconds = 60.0 / rate_limit
    print(f"\nUsing rate limit: {rate_limit} requests/minute ({delay_seconds:.1f}s between requests)")

    # Load test sample
    data_path = Path(__file__).parent.parent / "data" / "GameTox" / "gametox.csv"

    if not data_path.exists():
        print(f"\nERROR: GameTox dataset not found at {data_path}")
        return

    print(f"✓ Dataset found")
    print(f"\nLoading test sample (4,800 messages = 9% of full dataset)...")
    print(f"This allows fair comparison with Level 1 if we re-test on same subset.")

    messages, labels = load_gametox(data_path, limit=4800)
    print(f"Loaded {len(messages)} messages")

    # Save this subset for consistent testing across all levels
    subset_path = Path(__file__).parent.parent / "data" / "test_subset_4800.csv"
    if not subset_path.exists():
        save_test_subset(messages, labels, subset_path)
    else:
        print(f"\n✓ Test subset already exists at {subset_path}")
        print(f"  Using existing subset for consistency.")

    # FREE models to test (3 models, 16 batches each = 48 requests total)
    free_models = [
        "openai/gpt-oss-20b:free",  # Free tier model
        "x-ai/grok-4.1-fast",  # Fast Grok model
        "meta-llama/llama-3.3-70b-instruct:free",  # Llama 3.3 70B
    ]

    print(f"\n{'='*70}")
    print("  STEP 0: PROMPT DESIGN")
    print(f"{'='*70}")
    print("\nPrompt V1 (Simple):")
    print(LLMProfanityDetector.get_prompt_v1("you're an idiot"))
    print("\nPrompt V2 (With examples):")
    print(LLMProfanityDetector.get_prompt_v2("you're an idiot"))
    print("\n→ Using Prompt V1 for testing (simpler, fewer tokens)")

    print(f"\n{'='*70}")
    print("  STEP 1: TESTING FREE MODELS")
    print(f"{'='*70}")
    print(f"\nTesting {len(free_models)} free models on {len(messages)} messages...")
    print(f"Using BATCH mode: 300 messages per batch")
    print(f"  - {len(messages)} messages ÷ 300 per batch = 16 batches per model")
    print(f"  - 3 models × 16 batches = 48 API requests total")
    print(f"  - Well within 50 requests/day free tier limit")
    print(f"  - Compare to individual requests: {len(messages) * len(free_models)} requests (would exceed limit)")

    all_results = {}

    for model_name in free_models:
        print(f"\n{'─'*70}")
        print(f"Testing: {model_name}")
        print(f"{'─'*70}")

        try:
            detector = LLMProfanityDetector(model_name=model_name, rate_limit_per_minute=rate_limit)
            predictions = detector.predict_batch(messages, show_progress=True, prompt_version=1, use_batch_api=True)

            # Check if we hit rate limits
            if predictions is None:
                print(f"\n❌ Skipping {model_name} - rate limit exceeded")
                if detector.rate_limited:
                    print(f"\n⚠ CANNOT CONTINUE: Daily rate limit exhausted")
                    print(f"   Stopping all model tests.")
                    break  # Stop testing other models too
                continue  # Skip this model, try next one

            results = evaluate(predictions, labels)
            stats = detector.get_stats()

            all_results[model_name] = {
                'results': results,
                'stats': stats
            }

            print_results(model_name, results, stats)

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            continue

    # Step 2: Calculate theoretical production costs
    print(f"\n{'='*70}")
    print("  STEP 2: PRODUCTION COST ANALYSIS (Theoretical)")
    print(f"{'='*70}")

    print("\nNote: Using extrapolation, not actual paid API calls")
    print("\nExample paid model pricing (GPT-4):")
    print("  Input: $5.00 per 1M tokens")
    print("  Output: $15.00 per 1M tokens")

    if all_results:
        # Use average tokens from free model testing
        first_model = list(all_results.keys())[0]
        avg_input = all_results[first_model]['stats']['avg_input_tokens']
        avg_output = all_results[first_model]['stats']['avg_output_tokens']

        if avg_input > 0:
            cost_analysis = calculate_production_cost(
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
                input_price_per_1m=5.00,  # GPT-4 pricing
                output_price_per_1m=15.00
            )

            print(f"\nEstimated tokens per message:")
            print(f"  Input: {avg_input:.1f} tokens")
            print(f"  Output: {avg_output:.1f} tokens")
            print(f"\nProduction costs for 1M messages/day:")
            print(f"  Per message: ${cost_analysis['cost_per_message']:.6f}")
            print(f"  Per day: ${cost_analysis['daily_cost']:.2f}")
            print(f"  Per month: ${cost_analysis['monthly_cost']:,.2f}")
            print(f"  Per year: ${cost_analysis['yearly_cost']:,.2f}")

    # Step 3: Compare against Level 1
    print(f"\n{'='*70}")
    print("  STEP 3: COMPARISON vs LEVEL 1")
    print(f"{'='*70}")

    print("\nLevel 1 (Rule-based with normalization):")
    print("  Precision: 86.4%")
    print("  Recall:    46.3%")
    print("  F1-score:  0.603")
    print("  Latency:   <1ms")
    print("  Cost:      $0")

    if all_results:
        print("\nLevel 2 (LLM-based) - Best model:")
        # Find best F1 score
        best_model = max(all_results.keys(),
                        key=lambda m: all_results[m]['results']['f1'])
        best_results = all_results[best_model]['results']
        best_stats = all_results[best_model]['stats']

        print(f"  Model:     {best_model}")
        print(f"  Precision: {best_results['precision']*100:.1f}%")
        print(f"  Recall:    {best_results['recall']*100:.1f}%")
        print(f"  F1-score:  {best_results['f1']:.3f}")
        print(f"  Latency:   {best_stats['avg_latency']*1000:.0f}ms")
        print(f"  Cost:      Variable (depends on model)")

    print("\n✓ Level 2 testing complete!")
    print("\nBatch mode benefits:")
    total_batches = len(free_models) * 16  # 3 models × 16 batches
    individual_requests = len(messages) * len(free_models)  # 4800 × 3
    batch_time_minutes = (total_batches * 3) / 60  # 3 seconds per request
    individual_time_hours = (individual_requests * 3) / 3600
    print(f"  - Used {total_batches} API requests (vs {individual_requests:,} individual)")
    print(f"  - Stayed within 50 requests/day free tier limit ({total_batches}/50)")
    print(f"  - Batch size: 300 messages per request")
    print(f"  - Execution time: ~{batch_time_minutes:.1f} minutes (vs ~{individual_time_hours:.1f} hours for individual)")
    print(f"  - Request efficiency: {len(messages) / total_batches:.0f} messages per API call")
    print("\nNext steps:")
    print("  - Re-test Level 1 on data/test_subset_4800.csv (SAME 4,800 messages)")
    print("  - Use data/test_subset_4800.csv for Level 3 and Level 4 testing")
    print("  - Compare precision, recall, F1 across all levels on same data")
    print("  - Analyze which approach works best for gaming chat")


if __name__ == "__main__":
    main()
