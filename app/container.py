"""Khởi tạo các thành phần từ `Settings`. Tách riêng để test thay được từng phần."""

from dataclasses import dataclass

import httpx

from app.answer_cache import AnswerCache
from app.config import Settings
from app.embeddings import Embedder
from app.llm.gateway import ChatBackend, LLMGateway
from app.llm.openrouter import OpenRouterClient
from app.prompt_builder import PromptBuilder, load_knowledge
from app.quota import Quota
from app.router import IntentRouter, JevClassifier, build_router
from app.service import ChatService
from app.sessions import SessionStore
from app.store import KVStore, make_store
from app.usage_log import UsageLog


@dataclass
class Container:
    settings: Settings
    service: ChatService
    store: KVStore
    openrouter: OpenRouterClient
    router: IntentRouter
    usage_log: UsageLog

    async def aclose(self) -> None:
        for clf in self.router.classifiers:
            if isinstance(clf, JevClassifier):
                await clf.aclose()
        await self.openrouter.aclose()
        await self.store.aclose()
        self.usage_log.close()


def build_container(
    settings: Settings,
    *,
    http_client: httpx.AsyncClient | None = None,
    chat_backend: ChatBackend | None = None,
    router: IntentRouter | None = None,
    store: KVStore | None = None,
) -> Container:
    knowledge = load_knowledge(settings.knowledge_dir)
    prompt = PromptBuilder(knowledge, cache_control=settings.prompt_cache_control)
    openrouter = OpenRouterClient(settings, http_client=http_client)
    embedder = Embedder(openrouter, settings.embedding_model, timeout=settings.embedding_timeout_seconds)
    store = store or make_store(settings.redis_url)
    router = router or build_router(settings, embedder)
    usage_log = UsageLog(settings.usage_db_path)
    service = ChatService(
        settings,
        prompt=prompt,
        gateway=LLMGateway(settings, chat_backend or openrouter),
        router=router,
        answer_cache=AnswerCache(
            store,
            version=knowledge.version,
            ttl=settings.answer_cache_ttl_seconds,
            semantic_threshold=settings.semantic_cache_threshold,
            semantic_max_entries=settings.semantic_cache_max_entries,
        ),
        sessions=SessionStore(store, ttl=settings.session_ttl_seconds),
        quota=Quota(store, settings.plans, timezone=settings.quota_timezone),
        usage_log=usage_log,
        store=store,
        embedder=embedder,
    )
    return Container(settings, service, store, openrouter, router, usage_log)
