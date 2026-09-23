"""Cấu hình, đọc từ biến môi trường hoặc `.env`.

Biến nào cũng có giá trị (xem `.env.example`). Chỉ `OPENROUTER_API_KEY` không có mặc định: trống hoặc còn
`CHANGE_ME` thì dừng khi khởi động.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
PLACEHOLDER = "CHANGE_ME"


class ModelTarget(BaseModel):
    model: str
    # Slug nhà cung cấp trên OpenRouter, thử theo thứ tự. Rỗng thì OpenRouter tự chọn.
    providers: list[str] = Field(default_factory=list)


class _UnquotedEnvSource(EnvSettingsSource):
    """Bỏ một lớp nháy bao ngoài giá trị. `docker run --env-file` giữ nguyên nháy, `.env` và compose thì không."""

    def prepare_field_value(self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool) -> Any:
        if isinstance(value, str) and len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        return super().prepare_field_value(field_name, field, value, value_is_complex)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenRouter (model trả lời, Jev, embedding) ---
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_app_url: str = "http://localhost:8000"  # header HTTP-Referer
    openrouter_app_name: str = "CAG Chinese Tutor"  # header X-Title
    openrouter_data_collection: str = "allow"  # allow | deny
    prompt_cache_control: bool = False  # bật khi một tầng dùng Claude/Gemini
    llm_timeout_seconds: float = 60.0
    llm_connect_timeout_seconds: float = 5.0

    # --- Model theo tầng: phần tử sau là dự phòng ---
    tier_small: list[ModelTarget] = [
        ModelTarget(model="qwen/qwen3.8-flash", providers=["alibaba"]),
        ModelTarget(model="deepseek/deepseek-v4-flash"),
    ]
    tier_large: list[ModelTarget] = [
        ModelTarget(model="qwen/qwen3.7-plus", providers=["alibaba"]),
        ModelTarget(model="deepseek/deepseek-v4-pro"),
    ]
    reasoning_intents: list[str] = ["grammar"]  # intent của tầng large được bật thinking
    reasoning_max_tokens: int = 1024

    # --- Router: jev | embedding | none ---
    router_backend: str = "jev"
    router_fallback: str = "embedding"
    jev_model: str = "typesafe/jev-1.13"  # ghim version, ngưỡng confidence tune theo version
    jev_url: str = "https://openrouter.ai/api/v1/systemone"
    jev_timeout_seconds: float = 0.8
    jev_confidence_threshold: float = 0.6
    embedding_confidence_threshold: float = 0.55
    needs_context_threshold: float = 0.5

    # --- Embedding + cache gần giống ---
    embedding_model: str = "baai/bge-m3"
    embedding_timeout_seconds: float = 3.0
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_max_entries: int = 50_000

    # --- Lưu trữ: redis://... hoặc memory:// (chỉ cho dev) ---
    redis_url: str = "redis://localhost:6379/0"
    usage_db_path: Path = ROOT_DIR / "data" / "usage.sqlite3"
    answer_cache_ttl_seconds: int = 30 * 24 * 3600
    session_ttl_seconds: int = 24 * 3600
    max_turns_per_session: int = 10

    knowledge_dir: Path = ROOT_DIR / "knowledge"
    warmup_on_startup: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, _UnquotedEnvSource(settings_cls), dotenv_settings, file_secret_settings

    @field_validator("openrouter_api_key")
    @classmethod
    def _required(cls, value: str) -> str:
        value = value.strip()
        if not value or value == PLACEHOLDER:
            raise ValueError(f"OPENROUTER_API_KEY bắt buộc phải điền (đang trống hoặc còn là {PLACEHOLDER})")
        return value

    @field_validator("openrouter_data_collection")
    @classmethod
    def _data_collection(cls, value: str) -> str:
        if value not in ("allow", "deny"):
            raise ValueError("OPENROUTER_DATA_COLLECTION phải là allow hoặc deny")
        return value

    @field_validator("router_backend", "router_fallback")
    @classmethod
    def _router_backend(cls, value: str) -> str:
        if value not in ("jev", "embedding", "none"):
            raise ValueError("ROUTER_BACKEND/ROUTER_FALLBACK phải là jev, embedding hoặc none")
        return value

    @field_validator("redis_url")
    @classmethod
    def _redis_url(cls, value: str) -> str:
        if not value.startswith(("redis://", "rediss://", "unix://", "memory://")):
            raise ValueError("REDIS_URL phải là redis://, rediss://, unix:// hoặc memory://")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
