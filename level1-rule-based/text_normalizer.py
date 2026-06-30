#!/usr/bin/env python3
"""
Text Normalizer for Profanity Detection
Handles common evasion techniques: leetspeak, spacing, character insertion,
Unicode homoglyphs, repeated characters, and partial masking.
"""

import re
import unicodedata


class TextNormalizer:
    """
    Normalizes text to detect profanity bypass attempts.
    Applies multiple normalization strategies to catch evasion techniques.
    """

    def __init__(self):
        # Leetspeak substitution map (character → letter)
        self.leetspeak_map = {
            '0': 'o',
            '1': 'i',
            '3': 'e',
            '4': 'a',
            '5': 's',
            '7': 't',
            '8': 'b',
            '@': 'a',
            '$': 's',
            '!': 'i',
            '|': 'l',
            '(': 'c',
            ')': 'c',
            '<': 'c',
            '>': 'c',
            '+': 't',
            '&': 'and',
        }

        # Common homoglyphs (lookalike characters from different Unicode blocks)
        # Cyrillic → Latin
        self.homoglyph_map = {
            # Cyrillic lowercase
            'а': 'a',  # U+0430 (Cyrillic)
            'е': 'e',  # U+0435
            'о': 'o',  # U+043E
            'р': 'p',  # U+0440
            'с': 'c',  # U+0441
            'у': 'y',  # U+0443
            'х': 'x',  # U+0445
            'ѕ': 's',  # U+0455
            'і': 'i',  # U+0456
            'ј': 'j',  # U+0458
            'һ': 'h',  # U+04BB
            # Cyrillic uppercase
            'А': 'A',  # U+0410
            'В': 'B',  # U+0412
            'Е': 'E',  # U+0415
            'К': 'K',  # U+041A
            'М': 'M',  # U+041C
            'Н': 'H',  # U+041D
            'О': 'O',  # U+041E
            'Р': 'P',  # U+0420
            'С': 'C',  # U+0421
            'Т': 'T',  # U+0422
            'Х': 'X',  # U+0425
            # Greek
            'α': 'a',  # U+03B1
            'ο': 'o',  # U+03BF
            'ν': 'v',  # U+03BD
            'ρ': 'p',  # U+03C1
            # Mathematical bold/italic/etc
            '𝐚': 'a', '𝐛': 'b', '𝐜': 'c', '𝐝': 'd', '𝐞': 'e',
            '𝐟': 'f', '𝐠': 'g', '𝐡': 'h', '𝐢': 'i', '𝐣': 'j',
        }

        # Characters to strip (used for spacing/insertion evasion)
        self.strip_chars = r'[\s\-_.*|/\\,;:\'"`~+=\[\]{}()<>]'

    def normalize(self, text, aggressive=True):
        """
        Normalize text to detect evasion attempts.

        Args:
            text: Input text to normalize
            aggressive: If True, apply all normalizations. If False, only basic ones.

        Returns:
            Normalized text string
        """
        if not text:
            return text

        normalized = text.lower()

        # 1. Unicode normalization (NFKD decomposition)
        normalized = unicodedata.normalize('NFKD', normalized)

        # 2. Replace homoglyphs (Cyrillic/Greek lookalikes → Latin)
        normalized = self._replace_homoglyphs(normalized)

        # 3. Decode leetspeak (1 → i, 3 → e, $ → s, etc.)
        normalized = self._decode_leetspeak(normalized)

        if aggressive:
            # 4. Remove spacing and special character insertions
            # "f u c k" → "fuck", "f-u-c-k" → "fuck"
            normalized = re.sub(self.strip_chars, '', normalized)

        # 5. Collapse repeated characters
        # "fuuuuck" → "fuck", "shiiiit" → "shit"
        normalized = self._collapse_repeats(normalized)

        return normalized

    def _replace_homoglyphs(self, text):
        """Replace Unicode homoglyphs with ASCII equivalents."""
        for homoglyph, replacement in self.homoglyph_map.items():
            text = text.replace(homoglyph, replacement)
        return text

    def _decode_leetspeak(self, text):
        """Decode leetspeak substitutions."""
        for leet_char, replacement in self.leetspeak_map.items():
            text = text.replace(leet_char, replacement)
        return text

    def _collapse_repeats(self, text):
        """
        Collapse repeated characters.
        "fuuuuck" → "fuck", "heeeeey" → "hey"
        Keeps maximum 2 repeats to avoid over-normalization (e.g., "book" stays "book")
        """
        # Replace 3+ consecutive identical characters with just 1
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def normalize_preserving_boundaries(self, text):
        """
        Normalize text while preserving word boundaries (spaces).
        This allows regex patterns with \b to still work correctly.

        Strategy:
        1. Split text into words
        2. Normalize each word individually
        3. Rejoin with spaces

        This catches leetspeak/homoglyphs/repeats WITHIN words
        while keeping word boundaries intact for regex matching.
        """
        if not text:
            return text

        # Split on whitespace to preserve word boundaries
        words = text.split()

        normalized_words = []
        for word in words:
            # Normalize each word individually
            normalized_word = word.lower()

            # 1. Unicode normalization
            normalized_word = unicodedata.normalize('NFKD', normalized_word)

            # 2. Replace homoglyphs
            normalized_word = self._replace_homoglyphs(normalized_word)

            # 3. Decode leetspeak
            normalized_word = self._decode_leetspeak(normalized_word)

            # 4. Remove non-alphanumeric characters within the word
            # This catches "f-u-c-k" → "fuck", "s_h_i_t" → "shit"
            normalized_word = re.sub(r'[^a-z0-9]', '', normalized_word)

            # 5. Collapse repeated characters
            normalized_word = self._collapse_repeats(normalized_word)

            normalized_words.append(normalized_word)

        # Rejoin with spaces to preserve word boundaries
        return ' '.join(normalized_words)

    def generate_variants(self, text):
        """
        Generate multiple normalized variants for matching.
        Returns list of normalized strings with different aggressiveness levels.
        """
        variants = set()

        # Original text (lowercased)
        variants.add(text.lower())

        # Basic normalization (no aggressive stripping)
        variants.add(self.normalize(text, aggressive=False))

        # Aggressive normalization (strip everything)
        variants.add(self.normalize(text, aggressive=True))

        return list(variants)

    def detect_partial_masking(self, text, profane_words):
        """
        Detect partial masking like "f***", "s***", "b****".

        Args:
            text: Input text
            profane_words: List of profane words to check against

        Returns:
            True if partial masking is detected, False otherwise
        """
        # Pattern: word boundary + 1-3 letters + 2 or more asterisks
        partial_patterns = re.findall(r'\b(\w{1,3})\*{2,}', text.lower())

        for prefix in partial_patterns:
            # Check if any profane word starts with this prefix
            for word in profane_words:
                if word.startswith(prefix):
                    return True

        return False


def demo():
    """Demonstrate the normalizer on bypass examples."""
    normalizer = TextNormalizer()

    test_cases = [
        # Leetspeak
        ("sh1t", "shit"),
        ("a$$", "ass"),
        ("f*ck", "fck"),  # * gets decoded to empty, not 'u'
        ("h3ll", "hell"),
        ("1d10t", "idiot"),

        # Spacing
        ("f u c k", "fuck"),
        ("s h i t", "shit"),

        # Character insertion
        ("f-u-c-k", "fuck"),
        ("s_h_i_t", "shit"),
        ("d.a.m.n", "damn"),

        # Homoglyphs
        ("аss", "ass"),  # Cyrillic 'а'
        ("fuсk", "fuck"),  # Cyrillic 'с'
        ("һell", "hell"),  # Cyrillic 'һ'

        # Repeated characters
        ("fuuuuck", "fuck"),
        ("shiiiit", "shit"),
        ("damnnn", "damn"),
    ]

    print("="*70)
    print("  TEXT NORMALIZER DEMONSTRATION")
    print("="*70)
    print()
    print("Testing normalize_preserving_boundaries() [Option 4]:")
    print(f"{'Original':<25} {'Normalized':<25} {'Expected':<20} {'Match?'}")
    print("-"*70)

    for original, expected in test_cases:
        normalized = normalizer.normalize_preserving_boundaries(original)
        match = "✓" if expected in normalized else "✗"
        print(f"{original:<25} {normalized:<25} {expected:<20} {match}")

    # Test with full sentences
    print()
    print("Testing full sentences:")
    print("-"*70)
    sentence_tests = [
        ("You're such an 1d10t", "you're such an idiot"),
        ("what the f u c k", "what the fuck"),
        ("That's shiiiit", "that's shit"),
        ("Go to h3ll", "go to hell"),
    ]

    for original, expected in sentence_tests:
        normalized = normalizer.normalize_preserving_boundaries(original)
        match = "✓" if expected == normalized else "✗"
        print(f"{original:<25} → {normalized:<25} {match}")

    print()
    print("Partial Masking Detection:")
    print("-"*70)
    profane_words = ['fuck', 'shit', 'bitch', 'damn', 'ass']
    masking_tests = ["f***", "s***", "b****", "d***", "hello"]

    for test in masking_tests:
        detected = normalizer.detect_partial_masking(test, profane_words)
        status = "🚫 DETECTED" if detected else "✓ CLEAN"
        print(f"{test:<20} {status}")


if __name__ == "__main__":
    demo()
