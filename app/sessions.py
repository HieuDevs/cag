"""Lịch sử hội thoại theo phiên.

Chỉ nối thêm vào cuối, không cắt kiểu cửa sổ trượt: mỗi lần phần đầu `messages` đổi là mất cache
lịch sử. Quá `max_turns` lượt thì tóm tắt và mở phiên mới (xem docs/caching.md mục 2).
"""

import json
import uuid
from dataclasses import asdict, dataclass, field

from app.store import KVStore


@dataclass
class Session:
    id: str
    user_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    # Tóm tắt phiên trước, chỉ dùng cho lượt đầu tiên của phiên này.
    summary: str | None = None
    previous_id: str | None = None

    @property
    def turns(self) -> int:
        return sum(1 for m in self.messages if m["role"] == "user")

    @property
    def is_first_turn(self) -> bool:
        return not self.messages


class SessionStore:
    def __init__(self, store: KVStore, *, ttl: int):
        self.store = store
        self.ttl = ttl

    @staticmethod
    def _key(user_id: str, session_id: str) -> str:
        # Key có user_id: user không đọc được phiên của người khác dù biết session_id.
        return f"session:{user_id}:{session_id}"

    @staticmethod
    def new(user_id: str, *, summary: str | None = None, previous_id: str | None = None) -> Session:
        return Session(id=uuid.uuid4().hex, user_id=user_id, summary=summary, previous_id=previous_id)

    async def get(self, user_id: str, session_id: str) -> Session | None:
        raw = await self.store.get(self._key(user_id, session_id))
        return Session(**json.loads(raw)) if raw else None

    async def save(self, session: Session) -> None:
        await self.store.set(
            self._key(session.user_id, session.id), json.dumps(asdict(session), ensure_ascii=False), self.ttl
        )
