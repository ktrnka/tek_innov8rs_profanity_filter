"""
Gemini-based profanity filter — EXPERIMENTAL PROTOTYPE.

Why this exists: the Summer 2026 redesign (see COURSE_REDESIGN_2026.md) wants to know whether
Google's Gemini free tier (~1,500 req/day, no credit card) is a better default for the LLM level
than OpenRouter (50 req/day free). This module mirrors llm_filter.py (the OpenRouter version) so
the two can be compared directly on: (1) how hard the SDK setup is, (2) structured-output support,
and (3) whether safety filters refuse profanity *input*.

It is intentionally self-contained (no import from llm_filter) so the experiment doesn't depend on
the OpenRouter code path. If Gemini wins, this can be promoted/merged into the L3 reference.
"""

import os
from typing import Literal

import click
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


class ProfanityClassification(BaseModel):
    """Response format to force the LLM to generate a particular structure."""
    classification: Literal["profane", "clean"]


# The four adjustable text-harm categories, all set to BLOCK_NONE.
# Reasoning: this is a content-MODERATION task, so the *input* is expected to contain profanity.
# We must not let the safety filters block the input, or the classifier can't do its job. We test
# whether this is sufficient (question 3) rather than assuming it.
_SAFETY_OFF = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


class BlockedBySafetyError(RuntimeError):
    """Raised when Gemini refuses/blocks the request instead of classifying it.

    This is the failure mode we most care about for a moderation task — surface it loudly rather
    than silently treating a refusal as 'clean'.
    """


class GeminiProfanityFilter:
    """A Gemini-based profanity filter (prototype counterpart to LLMProfanityFilter)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        debug: bool = False,
    ):
        """Initialize the filter.

        Args:
            model: Gemini model ID. Default "gemini-2.5-flash-lite" — best free tier (30 RPM,
                1,500 req/day) and plenty for binary classification. Note "gemini-2.5-flash" is
                only 5 RPM on the free tier, and "gemini-2.0-flash" is deprecated (June 2026).
            api_key: API key; if None, loads GEMINI_API_KEY or GOOGLE_API_KEY from env/.env.
            debug: If True, print the full response + safety feedback for inspection.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env "
                "file. Get a free key at https://aistudio.google.com/apikey"
            )

        self.debug = debug
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

        # Same wording as llm_filter.py so prompt quality isn't a confound when comparing providers.
        self.system_prompt = (
            "You are a content moderation assistant. "
            "Classify the given text as either 'profane' or 'clean'. "
            "Text is profane if it contains offensive language, hate speech, "
            "harassment, or toxicity. "
            "Respond with only a single word: 'profane' or 'clean'."
        )

    def classify(self, text: str) -> bool:
        """Classify a single text as profane (True) or clean (False).

        Raises:
            BlockedBySafetyError: if Gemini blocked the request instead of returning a label.
        """
        response = self.client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.1,                       # low temp for deterministic-ish output
                response_mime_type="application/json",  # structured output
                response_schema=ProfanityClassification,
                safety_settings=_SAFETY_OFF,
            ),
        )

        if self.debug:
            click.echo(f"DEBUG: prompt_feedback: {response.prompt_feedback}", err=True)
            if response.candidates:
                click.echo(
                    f"DEBUG: finish_reason: {response.candidates[0].finish_reason}", err=True
                )
                click.echo(
                    f"DEBUG: safety_ratings: {response.candidates[0].safety_ratings}", err=True
                )
            click.echo(f"DEBUG: usage: {response.usage_metadata}", err=True)

        parsed = response.parsed
        if parsed is None:
            # No structured result — most likely a safety block on the prompt or the response.
            feedback = getattr(response, "prompt_feedback", None)
            finish = response.candidates[0].finish_reason if response.candidates else None
            raise BlockedBySafetyError(
                f"Gemini returned no parsed classification "
                f"(finish_reason={finish}, prompt_feedback={feedback}). "
                "This usually means a safety block — the key question for this prototype."
            )

        return parsed.classification == "profane"


@click.group()
def gemini():
    """Gemini-based filter (experimental — Level 3 provider comparison)."""
    pass


@gemini.command()
@click.argument("text")
@click.option("--model", "-m", default="gemini-2.5-flash-lite", help="Gemini model ID to use")
@click.option("--debug", is_flag=True, help="Show full API response + safety feedback")
def predict(text: str, model: str, debug: bool):
    """Test the Gemini filter on a single text.

    Usage:
        uv run main.py gemini predict "this is a test"
        uv run main.py gemini predict "you piece of shit" --debug
        uv run main.py gemini predict "hello" --model gemini-2.5-flash
    """
    filter_obj = GeminiProfanityFilter(model=model, debug=debug)

    try:
        result = filter_obj.classify(text)
    except BlockedBySafetyError as e:
        click.echo(f"Text: {text}")
        click.echo(f"Model: {model}")
        click.echo(f"Result: BLOCKED — {e}", err=True)
        raise SystemExit(2)

    click.echo(f"Text: {text}")
    click.echo(f"Model: {model}")
    click.echo(f"Result: {'PROFANE' if result else 'CLEAN'}")
