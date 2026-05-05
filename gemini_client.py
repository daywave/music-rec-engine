"""Shared Gemini client for all features."""

from google import genai

from config import GEMINI_API_KEY

_client = None


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def generate(prompt: str, system: str = "", max_tokens: int = 1000) -> str:
    """Generate text with Gemini."""
    client = get_gemini_client()
    config = {"max_output_tokens": max_tokens}
    if system:
        config["system_instruction"] = system
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )
    return response.text
