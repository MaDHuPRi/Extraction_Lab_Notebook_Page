"""
Ollama Client
-------------
Thin wrapper around the Ollama REST API.
Handles both vision (multimodal) and text-only requests.
"""

import requests
import json
from typing import Optional


class OllamaClient:
    """
    Client for the local Ollama API.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        vision_timeout: Timeout for vision model calls (slow, especially first call)
        text_timeout:   Timeout for text-only calls (faster)
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 vision_timeout: int = 300,   # 5 min — minicpm-v needs time
                 text_timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.vision_timeout = vision_timeout
        self.text_timeout = text_timeout

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list:
        """Return list of available model names."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m['name'] for m in r.json().get('models', [])]
        except Exception:
            return []

    def model_exists(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        available = self.list_models()
        return any(model_name in m or m in model_name for m in available)

    def chat(self,
             model: str,
             prompt: str,
             system: Optional[str] = None,
             temperature: float = 0.1) -> str:
        """
        Text-only chat request (Stage 4 assembly).
        Uses text_timeout — these calls are faster.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.text_timeout
        )
        response.raise_for_status()
        return response.json()['message']['content']

    def chat_with_image(self,
                         model: str,
                         prompt: str,
                         image_b64: str,
                         system: Optional[str] = None,
                         temperature: float = 0.1) -> str:
        """
        Multimodal chat request with image (Stage 1 classify + Stage 2 extract).
        Uses vision_timeout — minicpm-v can be slow, especially on first load.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        })

        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature},
            "stream": False
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.vision_timeout
        )
        response.raise_for_status()
        return response.json()['message']['content']

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama hub. Blocks until complete."""
        print(f"Pulling model '{model_name}'...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=3600
            )
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get('status', '')
                    if 'total' in data and 'completed' in data:
                        pct = data['completed'] / data['total'] * 100
                        print(f"\r  {status}: {pct:.1f}%", end='', flush=True)
                    elif status == 'success':
                        print(f"\n  Pull complete: {model_name}")
                        return True
            return True
        except Exception as e:
            print(f"\n  Pull failed: {e}")
            return False


    def warmup(self, vision_model: str, text_model: str):
        """
        Send a tiny dummy request to each model to force Ollama to load
        them into memory before the real pipeline calls start.
        First call is always slow (model loading); subsequent calls are fast.
        """
        print(f"[Warmup] Loading {vision_model} into memory...")
        try:
            # Tiny 4x4 white JPEG in base64
            white_pixel = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAEAAQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="
            self.chat_with_image(
                model=vision_model,
                prompt="Say OK",
                image_b64=white_pixel,
                temperature=0.0
            )
            print(f"[Warmup] {vision_model} ready.")
        except Exception as e:
            print(f"[Warmup] {vision_model} warmup failed (ok, continuing): {e}")

        print(f"[Warmup] Loading {text_model} into memory...")
        try:
            self.chat(model=text_model, prompt="Say OK", temperature=0.0)
            print(f"[Warmup] {text_model} ready.")
        except Exception as e:
            print(f"[Warmup] {text_model} warmup failed (ok, continuing): {e}")
