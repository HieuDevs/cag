"""Cấu hình ứng dụng, đọc từ biến môi trường hoặc file `.env`.

Mọi model đều gọi qua OpenRouter, nên ID model có dạng `<hãng>/<model>` như trên openrouter.ai/models.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class ModelTarget(BaseModel):
    """Một lựa chọn trong danh sách ưu tiên của một tầng."""

    model: str
    # Danh sách slug nhà cung cấp trên OpenRouter, thử theo thứ tự. Ghim nhà cung cấp giúp cache
    # phần đầu prompt luôn nằm ở một chỗ. Để trống thì OpenRouter tự chọn.
    providers: list[str] = Field(default_factory=list)


class PlanLimits(BaseModel):
    daily_requests: int
    allow_large: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Header tùy chọn để ứng dụng hiện trên bảng xếp hạng của OpenRouter.
    openrouter_app_url: str = ""
    openrouter_app_name: str = "CAG Chinese Tutor"
    # "deny" chỉ gửi tới nhà cung cấp không lưu dữ liệu để train. Xem câu hỏi mở về tuân thủ trong roadmap.
    openrouter_data_collection: str | None = None
    # Gửi `cache_control` trên phần system. Qwen và DeepSeek tự cache nên không cần, bật khi dùng
    # Claude hoặc Gemini cho một tầng.
    prompt_cache_control: bool = False
    llm_timeout_seconds: float = 60.0
    llm_connect_timeout_seconds: float = 5.0

    # tầng -> danh sách ưu tiên; phần tử đứng sau là dự phòng.
    tier_small: list[ModelTarget] = [
        ModelTarget(model="qwen/qwen3.8-flash", providers=["alibaba"]),
        ModelTarget(model="deepseek/deepseek-v4-flash"),
    ]
    tier_large: list[ModelTarget] = [
        ModelTarget(model="qwen/qwen3.7-plus", providers=["alibaba"]),
        ModelTarget(model="deepseek/deepseek-v4-pro"),
    ]
    # Intent nào của tầng `large` được bật thinking. Tầng `small` luôn tắt.
    reasoning_intents: list[str] = ["grammar"]
    reasoning_max_tokens: int = 1024

    # --- Router ---
    # "jev": Jev làm router chính, embedding làm dự phòng. "embedding": chỉ dùng embedding.
    # "none": không định tuyến, mọi câu hỏi vào `large`.
    router_backend: str = "jev"
    router_fallback: str = "embedding"
    typesafe_api_key: str = ""
    jev_model: str = "jev-1.13.0"
    jev_timeout_seconds: float = 0.8
    jev_confidence_threshold: float = 0.6
    embedding_confidence_threshold: float = 0.55
    needs_context_threshold: float = 0.5

    # --- Embedding (router dự phòng + cache câu trả lời gần giống) ---
    embedding_model: str = "baai/bge-m3"
    embedding_timeout_seconds: float = 3.0
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_max_entries: int = 50_000

    # --- Lưu trữ ---
    # Để trống thì dùng bộ nhớ trong tiến trình (chỉ hợp cho dev và test).
    redis_url: str = ""
    usage_db_path: Path = ROOT_DIR / "data" / "usage.sqlite3"
    answer_cache_ttl_seconds: int = 30 * 24 * 3600
    session_ttl_seconds: int = 24 * 3600
    max_turns_per_session: int = 10

    # --- Kiến thức ---
    knowledge_dir: Path = ROOT_DIR / "knowledge"

    # --- Quota ---
    plans: dict[str, PlanLimits] = {
        "free": PlanLimits(daily_requests=30, allow_large=True),
        "pro": PlanLimits(daily_requests=300, allow_large=True),
    }
    quota_timezone: str = "Asia/Ho_Chi_Minh"

    # --- API ---
    # Service chạy sau backend chính. Khi đặt giá trị, mọi request phải có `Authorization: Bearer <key>`.
    api_key: str = ""
    warmup_on_startup: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
