"""
LLM-based profanity filter implementation (Level 2).
"""

import os
from typing import Literal
import click
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel


class ProfanityClassification(BaseModel):
    """Response format to force the LLM to generate a particular structure."""
    classification: Literal["profane", "clean"]


class LLMProfanityFilter:
    """An LLM-based profanity filter using OpenRouter."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        debug: bool = False,
    ):
        """Initialize the LLM filter.

        Args:
            model: Model ID to use (e.g., "openai/gpt-4o-mini", "meta-llama/llama-3.3-70b-instruct:free")
            api_key: OpenRouter API key (if None, loads from OPENROUTER_API_KEY env var)
        """
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in .env file or pass api_key parameter."
            )
        self.debug = debug
        self.model = model

        # Initialize OpenAI client with OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Default system prompt for profanity detection
        self.system_prompt = (
            "You are a content moderation assistant. "
            "Classify the given text as either 'profane' or 'clean'. "
            "Text is profane if it contains offensive language, hate speech, "
            "harassment, or toxicity. "
            "Respond with only a single word: 'profane' or 'clean'."
        )

    def classify(self, text: str) -> bool:
        """Classify a single text as profane or not.

        Args:
            text: The text to classify

        Returns:
            True if profane, False if clean
        """
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=100,  # Limit just in case
            temperature=0.1,  # Low temperature for deterministic output
            response_format=ProfanityClassification,  # Force structured response
        )

        if self.debug:
            click.echo(f"DEBUG: Full response: {completion}", err=True)
            click.echo(f"Completion usage: {completion.usage}", err=True)

        classification = completion.choices[0].message.parsed
        if classification.classification == "profane":
            return True
        else:
            return False


@click.group()
def llm():
    """LLM-based filter using OpenRouter (Level 2)."""
    pass


@llm.command()
@click.argument("text")
@click.option("--model", "-m", default="openai/gpt-4o-mini", help="Model ID to use")
@click.option("--debug", is_flag=True, help="Show full API response for debugging")
def predict(text: str, model: str, debug: bool):
    """Test the LLM filter on a single text.

    Usage:
        uv run main.py llm predict "this is a test"
        uv run main.py llm predict "you noob" --model meta-llama/llama-3.3-70b-instruct:free
        uv run main.py llm predict "test" --model openai/gpt-oss-20b:free --debug
    """
    filter_obj = LLMProfanityFilter(model=model, debug=debug)

    result = filter_obj.classify(text)

    click.echo(f"Text: {text}")
    click.echo(f"Model: {model}")
    click.echo(f"Result: {'PROFANE' if result else 'CLEAN'}")
