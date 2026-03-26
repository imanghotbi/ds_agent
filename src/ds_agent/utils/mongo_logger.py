from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from ds_agent.config import settings
from ds_agent.utils.logger import logger


class MongoLLMLogger:
    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        enabled: bool = True,
        timeout_ms: int = 5000,
    ) -> None:
        self.uri = uri
        self.database = database
        self.collection_name = collection
        self.enabled = enabled
        self.timeout_ms = timeout_ms
        self._client: Optional[AsyncIOMotorClient] = None
        self._collection: Optional[AsyncIOMotorCollection] = None

    def _normalize_token_usage(self, token_usage: Any) -> Dict[str, Any]:
        if isinstance(token_usage, dict):
            return deepcopy(token_usage)
        if token_usage is None:
            return {}
        if hasattr(token_usage, "model_dump"):
            return token_usage.model_dump()
        if hasattr(token_usage, "__dict__"):
            return dict(token_usage.__dict__)
        return {"value": token_usage}

    def _extract_token_usage(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {}

        usage = getattr(response, "usage_metadata", None)
        normalized = self._normalize_token_usage(usage)
        if normalized:
            return normalized

        metadata = getattr(response, "response_metadata", None) or {}
        for key in ("token_usage", "usage", "usage_metadata"):
            normalized = self._normalize_token_usage(metadata.get(key))
            if normalized:
                return normalized

        return {}

    async def get_collection(self) -> Optional[AsyncIOMotorCollection]:
        if not self.enabled:
            return None

        if self._collection is not None:
            return self._collection

        try:
            self._client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
            )
            self._collection = self._client[self.database][self.collection_name]
            await self._client.admin.command("ping")
            return self._collection
        except Exception as exc:
            logger.warning(f"MongoDB is unavailable, skipping persistence: {exc}")
            self._collection = None
            if self._client is not None:
                self._client.close()
                self._client = None
            return None

    async def store_llm_call_log(self, node_name: str, session_id: str, response: Any) -> None:
        collection = await self.get_collection()
        if collection is None:
            return

        response_metadata = getattr(response, "response_metadata", None) or {}
        document = {
            "node_name": node_name,
            "session_id": session_id,
            "model_name": response_metadata.get("model_name", "unknown"),
            **self._extract_token_usage(response),
        }

        try:
            await collection.insert_one(document)
        except Exception as exc:
            logger.warning(f"Failed to persist LLM metadata to MongoDB: {exc}")

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
        self._client = None
        self._collection = None


mongo_llm_logger = MongoLLMLogger(
    uri=settings.mongo_uri,
    database=settings.mongo_database,
    collection=settings.mongo_collection,
    enabled=settings.mongo_enabled,
    timeout_ms=settings.mongo_logs_timeout_ms,
)
