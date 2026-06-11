"""
Gemini prototype smoke test — answers the three redesign questions in one run:

  1. SDK setup difficulty  -> if this script runs at all, setup is trivial.
  2. Structured output     -> every case that returns a clean PROFANE/CLEAN proves response_schema works.
  3. Safety refusals       -> any case that reports BLOCKED is a refusal on profanity *input*.

It also prints per-call latency and toy accuracy to get a rough feel for quality vs. the
OpenRouter path (llm_filter.py). This is a throwaway experiment harness, not student material.

Run:  uv run python gemini_smoke_test.py   (needs GEMINI_API_KEY in .env)
"""

import time

from gemini_filter import GeminiProfanityFilter, BlockedBySafetyError

# (text, expected_profane). Spans: clean, gaming banter, strong profanity, a slur, sexual content,
# and the classic Scunthorpe-style false-positive traps (clean words containing profane substrings).
CASES: list[tuple[str, bool]] = [
    ("good game everyone, well played", False),
    ("how do i get to the next zone?", False),
    ("you are absolute trash at this game", True),
    ("get good you fucking noob", True),
    ("i will kill you in real life", True),
    ("you piece of shit, uninstall", True),
    ("kys loser", True),
    ("nice assessment of the situation", False),   # 'ass' substring, clean
    ("i live in scunthorpe", False),                # classic false-positive trap
    ("let's meet at the class reunion", False),     # 'ass' substring, clean
]


def main() -> None:
    print("Initializing GeminiProfanityFilter...")
    flt = GeminiProfanityFilter()
    print(f"Model: {flt.model}\n")

    correct = 0
    blocked = 0
    latencies: list[float] = []

    for text, expected in CASES:
        t0 = time.perf_counter()
        try:
            pred = flt.classify(text)
            dt = time.perf_counter() - t0
            latencies.append(dt)
            ok = "OK " if pred == expected else "XX "
            correct += pred == expected
            label = "PROFANE" if pred else "CLEAN"
            print(f"{ok} [{dt:5.2f}s] {label:7s} (exp {'PROFANE' if expected else 'CLEAN'})  {text!r}")
        except BlockedBySafetyError as e:
            dt = time.perf_counter() - t0
            blocked += 1
            print(f"!! [{dt:5.2f}s] BLOCKED  {text!r}\n      -> {e}")

    n = len(CASES)
    answered = n - blocked
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Q1 setup:            OK (script ran)")
    print(f"Q2 structured output: {answered}/{n} returned a parsed label (rest were blocked/errored)")
    print(f"Q3 safety refusals:   {blocked}/{n} blocked  <-- 0 is the result we want")
    if latencies:
        print(f"Latency:             avg {sum(latencies)/len(latencies):.2f}s, "
              f"min {min(latencies):.2f}s, max {max(latencies):.2f}s")
    if answered:
        print(f"Toy accuracy:        {correct}/{answered} on answered cases")


if __name__ == "__main__":
    main()
