"""Command-line interface for profanity filter."""

import sys
import argparse
from typing import Optional
import json

from .detector import ProfanityDetector


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Profanity Filter - Hybrid toxicity detection for gaming chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single message (auto mode - uses hybrid)
  profanity-filter check "your message here"

  # Check with specific model
  profanity-filter check "your message" --model modernbert-multiclass

  # Check from file
  profanity-filter check --file messages.txt

  # Hybrid mode with custom threshold
  profanity-filter check "your message" --mode hybrid --threshold 0.8

  # List available models
  profanity-filter list-models

  # Batch check with JSON output
  profanity-filter check --file messages.txt --json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check text for toxicity')
    check_parser.add_argument(
        'text',
        nargs='?',
        help='Text to check (or use --file)'
    )
    check_parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing messages (one per line)'
    )
    check_parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['traditional-ml', 'modernbert-binary', 'modernbert-multiclass', 'toxic-bert'],
        help='Specific model to use (overrides mode)'
    )
    check_parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'hybrid', 'auto'],
        default='auto',
        help='Detection mode (default: auto = hybrid)'
    )
    check_parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.7,
        help='Confidence threshold for hybrid mode (default: 0.7)'
    )
    check_parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    check_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with timing info'
    )

    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')

    args = parser.parse_args()

    if args.command == 'check':
        run_check(args)
    elif args.command == 'list-models':
        list_models()
    else:
        parser.print_help()
        sys.exit(1)


def run_check(args):
    """Run toxicity check."""
    # Get texts to check
    if args.file:
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        print("Error: Provide text as argument or use --file")
        sys.exit(1)

    # Initialize detector
    detector = ProfanityDetector(
        model=args.model,
        mode=args.mode,
        confidence_threshold=args.threshold
    )

    # Check texts
    if len(texts) == 1:
        result = detector.predict(texts[0])
        results = [result]
    else:
        results = detector.predict_batch(texts)

    # Output results
    if args.json:
        # JSON output
        output = []
        for result in results:
            output.append({
                'text': result.text,
                'is_toxic': result.is_toxic,
                'confidence': result.confidence,
                'toxicity_type': result.toxicity_type,
                'latency_ms': result.latency_ms,
                'model': result.model_name,
                'two_stage': result.two_stage
            })
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        for i, result in enumerate(results, 1):
            if len(texts) > 1:
                print(f"\n[{i}/{len(results)}] {result.text[:60]}...")
            else:
                print(f"\nText: {result.text}")

            # Status with color
            status = "🔴 TOXIC" if result.is_toxic else "🟢 CLEAN"
            print(f"Status: {status}")
            print(f"Confidence: {result.confidence:.1%}")

            if result.toxicity_type:
                print(f"Type: {result.toxicity_type}")

            if args.verbose:
                print(f"Model: {result.model_name}")
                print(f"Latency: {result.latency_ms:.2f}ms")
                if result.two_stage:
                    print("Mode: Two-stage hybrid filtering")

    # Summary for batch
    if len(texts) > 1 and not args.json:
        toxic_count = sum(1 for r in results if r.is_toxic)
        avg_latency = sum(r.latency_ms for r in results) / len(results)

        print(f"\n{'='*60}")
        print(f"Summary: {toxic_count}/{len(texts)} toxic ({toxic_count/len(texts):.1%})")
        print(f"Average latency: {avg_latency:.2f}ms per message")


def list_models():
    """List available models."""
    models = ProfanityDetector.list_models()

    print("\nAvailable Models:")
    print("=" * 60)

    for name, description in models.items():
        print(f"\n{name}")
        print(f"  {description}")

    print("\nPerformance Summary:")
    print("-" * 60)
    print("  traditional-ml:         F1=0.68 (GameTox), 0.008ms latency")
    print("  modernbert-binary:      F1=0.78 (GameTox), 14ms latency")
    print("  modernbert-multiclass:  F1=0.85 (GameTox), 16ms latency ⭐")
    print("  toxic-bert:             F1=0.67 (External), 10ms latency")

    print("\nRecommended:")
    print("  - Gaming chat: modernbert-multiclass (best in-domain)")
    print("  - General use: toxic-bert (best generalization)")
    print("  - High speed:  traditional-ml (1,700x faster)")
    print("  - Auto/hybrid: Combines fast + accurate (recommended)")


if __name__ == '__main__':
    main()
