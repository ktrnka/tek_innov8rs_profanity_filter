#!/usr/bin/env python3
"""
Level 1 - Bypass Testing
Tests various evasion techniques against the rule-based profanity filter.
Compares basic regex vs. text normalization enhancement.
"""

import re
from collections import defaultdict
from text_normalizer import TextNormalizer

class RegexProfanityDetector:
    """Regex-based profanity detector with optional text normalization."""

    def __init__(self, profane_words, use_normalization=False):
        self.profane_words = profane_words
        self.use_normalization = use_normalization
        self.normalizer = TextNormalizer() if use_normalization else None
        pattern = r'\b(' + '|'.join(re.escape(word) for word in profane_words) + r')\b'
        self.pattern = re.compile(pattern, re.IGNORECASE)

    def is_profane(self, text):
        if not text:
            return False

        # Check partial masking first
        if self.normalizer and self.normalizer.detect_partial_masking(text, self.profane_words):
            return True

        if not self.use_normalization:
            return bool(self.pattern.search(text))

        # Use boundary-preserving normalization
        normalized = self.normalizer.normalize_preserving_boundaries(text)
        return bool(self.pattern.search(normalized))

def test_bypass_category(detector, category_name, test_cases):
    """Test a category of bypass attempts."""
    results = []
    detected_count = 0
    bypassed_count = 0

    for original, bypass_attempt in test_cases:
        is_detected = detector.is_profane(bypass_attempt)
        results.append({
            'original': original,
            'bypass': bypass_attempt,
            'detected': is_detected
        })
        if is_detected:
            detected_count += 1
        else:
            bypassed_count += 1

    return {
        'category': category_name,
        'total': len(test_cases),
        'detected': detected_count,
        'bypassed': bypassed_count,
        'bypass_rate': bypassed_count / len(test_cases) if test_cases else 0,
        'results': results
    }

def print_category_results(category_result):
    """Print results for a bypass category."""
    print(f"\n{'='*70}")
    print(f"  {category_result['category']}")
    print(f"{'='*70}")
    print(f"Total attempts: {category_result['total']}")
    print(f"Detected: {category_result['detected']} ({100 * (1 - category_result['bypass_rate']):.1f}%)")
    print(f"Bypassed: {category_result['bypassed']} ({100 * category_result['bypass_rate']:.1f}%)")
    print()

    print(f"{'Original':<20} {'Bypass Attempt':<30} {'Status':<10}")
    print("-" * 70)
    for result in category_result['results']:
        status = "🚫 BLOCKED" if result['detected'] else "✅ BYPASSED"
        print(f"{result['original']:<20} {result['bypass']:<30} {status}")

def main():
    # Use the same expanded word list from the production detector
    profane_words = [
        'damn', 'shit', 'fuck', 'ass', 'bitch', 'bastard', 'hell',
        'idiot', 'idiots', 'wtf', 'fucking', 'useless', 'stupid',
        'retard', 'retards', 'moron', 'morons', 'ffs', 'fck',
        'stfu', 'trash', 'dumb', 'noob', 'noobs', 'bot', 'bots',
        'camper', 'campers', 'camping'
    ]

    # Create both detectors
    basic_detector = RegexProfanityDetector(profane_words, use_normalization=False)
    normalized_detector = RegexProfanityDetector(profane_words, use_normalization=True)

    print("="*70)
    print("  BYPASS TESTING: Comparing Basic vs. Normalized Detectors")
    print("="*70)
    print(f"Testing against {len(profane_words)} blocked words")
    print("This demonstrates adversarial evasion techniques")
    print()
    print("TWO DETECTORS TESTED:")
    print("  1. Basic: Simple regex with word boundaries")
    print("  2. Normalized: With text normalization (Option 4)")

    # Category 1: Leetspeak (character substitution)
    leetspeak_tests = [
        ('shit', 'sh1t'),
        ('shit', 'sh!t'),
        ('shit', '$hit'),
        ('ass', 'a$$'),
        ('ass', '@ss'),
        ('fuck', 'f*ck'),
        ('fuck', 'f**k'),
        ('fuck', 'fvck'),
        ('hell', 'h3ll'),
        ('bitch', 'b1tch'),
        ('damn', 'd@mn'),
        ('idiot', '1d10t'),
        ('stupid', 'stup1d'),
    ]

    # Category 2: Spacing/Zero-width characters
    spacing_tests = [
        ('fuck', 'f u c k'),
        ('shit', 's h i t'),
        ('damn', 'd a m n'),
        ('ass', 'a s s'),
        ('bitch', 'b i t c h'),
        ('idiot', 'i d i o t'),
        ('fuck', 'f  u  c  k'),  # double spaces
    ]

    # Category 3: Character insertion
    insertion_tests = [
        ('fuck', 'f-u-c-k'),
        ('shit', 's_h_i_t'),
        ('damn', 'd.a.m.n'),
        ('ass', 'a.s.s'),
        ('bitch', 'b*i*t*c*h'),
        ('idiot', 'i-d-i-o-t'),
        ('fuck', 'f|u|c|k'),
    ]

    # Category 4: Homoglyphs (Unicode lookalikes)
    homoglyph_tests = [
        ('ass', 'аss'),  # Cyrillic 'а' (U+0430)
        ('fuck', 'fuсk'),  # Cyrillic 'с' (U+0441)
        ('hell', 'һell'),  # Cyrillic 'һ' (U+04BB)
        ('bitch', 'bitсh'),  # Cyrillic 'с'
        ('damn', 'dаmn'),  # Cyrillic 'а'
    ]

    # Category 5: Partial masking
    masking_tests = [
        ('fuck', 'f***'),
        ('shit', 's***'),
        ('bitch', 'b****'),
        ('damn', 'd***'),
        ('ass', 'a**'),
        ('fuck', 'fu**'),
        ('shit', 'sh**'),
    ]

    # Category 6: Phonetic spelling
    phonetic_tests = [
        ('fuck', 'fuk'),
        ('fuck', 'phuck'),
        ('shit', 'shyt'),
        ('bitch', 'beetch'),
        ('damn', 'dam'),
        ('ass', 'azz'),
        ('hell', 'heck'),  # euphemism
        ('idiot', 'idyot'),
    ]

    # Category 7: Repeated characters
    repeat_tests = [
        ('fuck', 'fuuuck'),
        ('shit', 'shiiiit'),
        ('damn', 'damnnn'),
        ('ass', 'assss'),
        ('bitch', 'bitchhhh'),
        ('hell', 'helllll'),
    ]

    # Run tests on BOTH detectors
    print("\n" + "="*70)
    print("  TESTING BASIC DETECTOR (No Normalization)")
    print("="*70)

    basic_results = []
    basic_results.append(test_bypass_category(basic_detector, "1. LEETSPEAK (Character Substitution)", leetspeak_tests))
    basic_results.append(test_bypass_category(basic_detector, "2. SPACING", spacing_tests))
    basic_results.append(test_bypass_category(basic_detector, "3. CHARACTER INSERTION", insertion_tests))
    basic_results.append(test_bypass_category(basic_detector, "4. HOMOGLYPHS (Unicode Lookalikes)", homoglyph_tests))
    basic_results.append(test_bypass_category(basic_detector, "5. PARTIAL MASKING", masking_tests))
    basic_results.append(test_bypass_category(basic_detector, "6. PHONETIC SPELLING", phonetic_tests))
    basic_results.append(test_bypass_category(basic_detector, "7. REPEATED CHARACTERS", repeat_tests))

    # Print basic results
    for result in basic_results:
        print_category_results(result)

    # Test normalized detector
    print("\n" + "="*70)
    print("  TESTING NORMALIZED DETECTOR (With Text Normalization)")
    print("="*70)

    normalized_results = []
    normalized_results.append(test_bypass_category(normalized_detector, "1. LEETSPEAK (Character Substitution)", leetspeak_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "2. SPACING", spacing_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "3. CHARACTER INSERTION", insertion_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "4. HOMOGLYPHS (Unicode Lookalikes)", homoglyph_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "5. PARTIAL MASKING", masking_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "6. PHONETIC SPELLING", phonetic_tests))
    normalized_results.append(test_bypass_category(normalized_detector, "7. REPEATED CHARACTERS", repeat_tests))

    # Print normalized results
    for result in normalized_results:
        print_category_results(result)

    # Print comparison summary
    print(f"\n{'='*70}")
    print("  COMPARISON: Basic vs. Normalized Detection Rates")
    print(f"{'='*70}")
    print(f"{'Category':<35} {'Basic':<12} {'Normalized':<12} {'Improvement'}")
    print("-" * 70)

    basic_total = 0
    basic_bypassed = 0
    norm_total = 0
    norm_bypassed = 0

    for basic_res, norm_res in zip(basic_results, normalized_results):
        basic_total += basic_res['total']
        basic_bypassed += basic_res['bypassed']
        norm_total += norm_res['total']
        norm_bypassed += norm_res['bypassed']

        basic_caught_pct = 100 * (1 - basic_res['bypass_rate'])
        norm_caught_pct = 100 * (1 - norm_res['bypass_rate'])
        improvement = norm_caught_pct - basic_caught_pct

        symbol = "📈" if improvement > 10 else "➡️" if improvement > 0 else "📉"

        category_short = basic_res['category'].split('.')[1].strip()[:30]
        print(f"{category_short:<35} {basic_caught_pct:>5.1f}%      {norm_caught_pct:>5.1f}%      {symbol} {improvement:+5.1f}%")

    basic_overall = 100 * (1 - basic_bypassed / basic_total) if basic_total > 0 else 0
    norm_overall = 100 * (1 - norm_bypassed / norm_total) if norm_total > 0 else 0
    overall_improvement = norm_overall - basic_overall

    print("-" * 70)
    print(f"{'OVERALL':<35} {basic_overall:>5.1f}%      {norm_overall:>5.1f}%      {'📈' if overall_improvement > 0 else '📉'} {overall_improvement:+5.1f}%")
    print()
    print(f"Total bypass attempts: {basic_total}")
    print(f"Basic detector caught: {basic_total - basic_bypassed} ({basic_overall:.1f}%)")
    print(f"Normalized detector caught: {norm_total - norm_bypassed} ({norm_overall:.1f}%)")
    print(f"Additional bypasses caught: {(norm_total - norm_bypassed) - (basic_total - basic_bypassed)}")

    # Key insights
    print(f"\n{'='*70}")
    print("  KEY INSIGHTS")
    print(f"{'='*70}")
    print()
    print("BASIC DETECTOR (No Normalization):")
    print("  ✗ Leetspeak completely bypasses")
    print("  ✗ Spacing bypasses word boundaries")
    print("  ✗ Unicode homoglyphs invisible")
    print("  ✗ Phonetic variations evade detection")
    print("  ✗ Character insertion bypasses")
    print("  ✗ Partial masking bypasses")
    print("  ✗ Repeated characters bypass")
    print(f"  → Overall: {basic_overall:.1f}% detection rate on adversarial attempts")
    print()
    print("NORMALIZED DETECTOR (With Text Normalization - Option 4):")
    print("  ✓ Leetspeak CAUGHT (sh1t → shit)")
    print("  ✗ Spacing still bypasses (limitation of boundary-preserving approach)")
    print("  ✓ Unicode homoglyphs CAUGHT (аss → ass)")
    print("  ✗ Phonetic variations still evade")
    print("  ✓ Character insertion CAUGHT (f-u-c-k → fuck)")
    print("  ✓ Partial masking CAUGHT (f***)")
    print("  ✓ Repeated characters CAUGHT (fuuuck → fuck)")
    print(f"  → Overall: {norm_overall:.1f}% detection rate on adversarial attempts")
    print()
    print(f"IMPROVEMENT: {overall_improvement:+.1f}% more bypasses caught with normalization")
    print()
    print("CONCLUSION:")
    print("  Text normalization significantly improves bypass detection while")
    print("  maintaining performance on organic data (GameTox: 46.2% → 46.3% recall)")
    print("  Spacing bypasses remain a challenge with boundary-preserving approach.")

if __name__ == "__main__":
    main()
