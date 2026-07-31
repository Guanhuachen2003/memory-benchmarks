"""OpenRouter embedding provider for the self-hosted Mem0 server."""

import json
import logging
import os
from typing import Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase
from token_meter import record_usage

logger = logging.getLogger(__name__)


class OpenRouterEmbedding(EmbeddingBase):
    """Generate vectors with OpenRouter's OpenAI-compatible embeddings API."""

    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)
        self.api_key = (
            os.getenv("MEM0_EMBEDDING_API_KEY")
            or getattr(self.config, "api_key", None)
            or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("MEM0_EMBEDDING_API_KEY (or OPENAI_API_KEY) is required")

        self.url = os.getenv(
            "OPENROUTER_EMBEDDING_URL",
            "https://openrouter.ai/api/v1/embeddings",
        )
        self.model = self.config.model or "openai/text-embedding-3-small"
        self.dims = self.config.embedding_dims or 1536
        logger.info(
            "OpenRouter embedder: model=%s, dimensions=%d, endpoint=%s",
            self.model,
            self.dims,
            self.url,
        )

    def _request(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": [text.replace("\n", " ") for text in texts],
            "dimensions": self.dims,
            "encoding_format": "float",
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                result = json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"OpenRouter embeddings returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenRouter embeddings request failed: {exc.reason}") from exc

        record_usage("embedding", result.get("usage"))
        rows = sorted(result.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) for vector in vectors):
            raise RuntimeError("OpenRouter returned an invalid embeddings response")
        if any(len(vector) != self.dims for vector in vectors):
            raise RuntimeError(
                f"OpenRouter returned vectors that do not match configured dimensions ({self.dims})"
            )
        return vectors

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> list[float]:
        return self._request([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        memory_action: str = "add",
    ) -> list[list[float]]:
        return self._request(texts)
