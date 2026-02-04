"""
OpenRouter Client
Provides optional LLM-based object descriptions using an API key in env var.
"""

import os
import json
from typing import Optional

import requests


class OpenRouterClient:
    """Lightweight OpenRouter client for object descriptions"""

    def __init__(self, api_key_env: str = "OPENROUTER_API_KEY"):
        self.api_key = os.getenv(api_key_env)
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")
        self.site_url = os.getenv("OPENROUTER_SITE_URL", "")
        self.site_title = os.getenv("OPENROUTER_SITE_TITLE", "EchoVision AI")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def describe_object(self, object_name: str, context: Optional[str] = None) -> Optional[str]:
        """
        Ask OpenRouter for an object description.

        Returns None if no API key is available or request fails.
        """
        if not self.api_key:
            return None

        prompt = (
            f"Describe the object '{object_name}' in 1-2 short sentences. "
            "Focus on typical appearance and use."
        )
        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_title:
            headers["X-Title"] = self.site_title

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=20
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
