"""
Python client for the Invoice Processing Pipeline environment.

Usage:
    from client import InvoiceEnvClient
    from models import InvoiceAction

    client = InvoiceEnvClient(base_url="http://localhost:7860")
    result = client.reset(task_id="easy")
    print(result["observation"]["raw_text"])

    result = client.step({"vendor": "Acme Corp", "date": "2024-06-15", ...})
    print(result["reward"])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class InvoiceEnvClient:
    """Synchronous HTTP client for the Invoice Processing Pipeline."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        resp = self._client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, extracted_data: Dict[str, Any], explanation: str = "",
             episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit extracted/cleaned data and get reward + feedback."""
        body: Dict[str, Any] = {"extracted_data": extracted_data, "explanation": explanation}
        if episode_id is not None:
            body["episode_id"] = episode_id
        resp = self._client.post(f"{self.base_url}/step", json=body)
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        """Get current episode state."""
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        """List available tasks and schemas."""
        resp = self._client.get(f"{self.base_url}/tasks")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncInvoiceEnvClient:
    """Async HTTP client for the Invoice Processing Pipeline."""

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        resp = await self._client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    async def step(self, extracted_data: Dict[str, Any], explanation: str = "",
                   episode_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"extracted_data": extracted_data, "explanation": explanation}
        if episode_id is not None:
            body["episode_id"] = episode_id
        resp = await self._client.post(f"{self.base_url}/step", json=body)
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> Dict[str, Any]:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()