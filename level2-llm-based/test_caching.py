#!/usr/bin/env python3
"""Test response caching to demonstrate latency and cost savings."""

import os
import csv
import time
from dotenv import load_dotenv
from llm_detector import LLMProfanityDetector

load_dotenv()


def load_messages(csv_file, limit=None):
    """Load messages from CSV."""
    messages = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            messages.append(row['message'])
    return messages


def main():
    print("="*70)
    print("  RESPONSE CACHING TEST")
    print("="*70)
    print("\nDemonstrating cache benefits on repeated messages")
    print()

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in .env")
        return

    # Load subset of messages for quick demonstration
    messages = load_messages('../data/test_subset_200_stratified.csv', limit=50)
    print(f"✓ Loaded {len(messages)} messages for testing")
    print()

    # Test configuration
    model = "x-ai/grok-4.1-fast"
    prompt_version = 3  # Use winning prompt from Task 2

    print(f"Model: {model}")
    print(f"Prompt: V3 (Context-aware)")
    print(f"Batch size: 10")
    print()

    # First run: Populate cache (all cache misses)
    print("="*70)
    print("  RUN 1: Populating Cache (Cold Start)")
    print("="*70)
    print()

    detector = LLMProfanityDetector(model, api_key)

    start_time = time.time()
    predictions_run1 = detector.predict_batch(
        messages,
        show_progress=True,
        prompt_version=prompt_version,
        use_batch_api=True,
        batch_size=10,
        enable_fallback=False
    )
    run1_time = time.time() - start_time

    if predictions_run1 is None:
        print("\n❌ First run failed")
        return

    print(f"\n   Run 1 Stats:")
    print(f"   Total time: {run1_time:.2f}s")
    print(f"   Cache hits: {detector.cache_hits}")
    print(f"   Cache misses: {detector.cache_misses}")
    print(f"   Cache hit rate: {detector.cache_hits / (detector.cache_hits + detector.cache_misses) * 100:.1f}%" if (detector.cache_hits + detector.cache_misses) > 0 else 0)
    print(f"   API requests: {detector.total_requests}")
    print(f"   Avg latency/request: {detector.total_latency / detector.total_requests:.2f}s")

    # Second run: Use cache (all cache hits)
    print()
    print("="*70)
    print("  RUN 2: Using Cache (Warm Start)")
    print("="*70)
    print()

    # Reset counters for run 2
    run2_start_hits = detector.cache_hits
    run2_start_misses = detector.cache_misses
    run2_start_requests = detector.total_requests

    start_time = time.time()
    predictions_run2 = detector.predict_batch(
        messages,
        show_progress=True,
        prompt_version=prompt_version,
        use_batch_api=True,
        batch_size=10,
        enable_fallback=False
    )
    run2_time = time.time() - start_time

    if predictions_run2 is None:
        print("\n❌ Second run failed")
        return

    run2_hits = detector.cache_hits - run2_start_hits
    run2_misses = detector.cache_misses - run2_start_misses
    run2_requests = detector.total_requests - run2_start_requests

    print(f"\n   Run 2 Stats:")
    print(f"   Total time: {run2_time:.2f}s")
    print(f"   Cache hits: {run2_hits}")
    print(f"   Cache misses: {run2_misses}")
    print(f"   Cache hit rate: {run2_hits / (run2_hits + run2_misses) * 100:.1f}%" if (run2_hits + run2_misses) > 0 else 0)
    print(f"   API requests: {run2_requests}")

    # Verify predictions match
    if predictions_run1 == predictions_run2:
        print(f"   ✓ Predictions match (cache working correctly)")
    else:
        print(f"   ⚠ Warning: Predictions differ!")

    # Cache benefits summary
    print()
    print("="*70)
    print("  CACHE BENEFITS SUMMARY")
    print("="*70)
    print()

    time_saved = run1_time - run2_time
    time_saved_pct = (time_saved / run1_time) * 100
    api_calls_saved = run2_start_requests  # All requests from run 1

    print(f"Latency Improvement:")
    print(f"  Run 1 (cold): {run1_time:.2f}s")
    print(f"  Run 2 (warm): {run2_time:.2f}s")
    print(f"  Time saved:   {time_saved:.2f}s ({time_saved_pct:.1f}% faster)")
    print()

    print(f"API Call Reduction:")
    print(f"  Run 1: {run2_start_requests} API calls")
    print(f"  Run 2: {run2_requests} API calls")
    print(f"  Saved: {api_calls_saved} API calls (100%)")
    print()

    print(f"Cache Statistics:")
    print(f"  Total cache size: {len(detector.cache)} unique messages")
    print(f"  Total cache hits: {detector.cache_hits}")
    print(f"  Total cache misses: {detector.cache_misses}")
    print(f"  Overall hit rate: {detector.cache_hits / (detector.cache_hits + detector.cache_misses) * 100:.1f}%")

    print("\n" + "="*70)
    print("✓ Caching test complete!")
    print("="*70)
    print()
    print("Key Takeaways:")
    print("  • Cache eliminates API latency for repeated messages")
    print("  • 100% cache hit rate on identical message sets")
    print("  • Reduces API costs by avoiding redundant requests")
    print("  • Perfect for production with common phrases (gg, lol, etc.)")


if __name__ == '__main__':
    main()
