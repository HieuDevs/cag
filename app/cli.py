"""Lệnh quản trị.

    cag knowledge check     # CI: sửa knowledge/ mà chưa tăng KNOWLEDGE_VERSION thì báo lỗi
    cag knowledge bump      # tăng version (YYYY-MM-DD.N) và ghi lại checksum
    cag knowledge info      # số file, số ký tự, ước lượng token
    cag warmup              # làm nóng cache phần đầu prompt cho model chính của mỗi tầng
    cag stats [--hours 24]  # chỉ số từ log usage
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from app.config import ROOT_DIR, get_settings
from app.prompt_builder import CHECKSUM_FILE, VERSION_FILE, compute_checksum, load_knowledge


def _knowledge_dir() -> Path:
    # Lệnh `knowledge` chạy được trong CI mà không cần API key, nên không đọc toàn bộ Settings.
    return Path(os.environ.get("KNOWLEDGE_DIR") or ROOT_DIR / "knowledge")


def knowledge_check() -> int:
    d = _knowledge_dir()
    recorded = (d / CHECKSUM_FILE).read_text().strip() if (d / CHECKSUM_FILE).exists() else ""
    actual = compute_checksum(d)
    if recorded != actual:
        print(
            "knowledge/ đã thay đổi nhưng KNOWLEDGE_VERSION chưa tăng.\n"
            "Chạy `cag knowledge bump` rồi commit cả VERSION và CHECKSUM.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: version {(d / VERSION_FILE).read_text().strip()}, checksum {actual[:12]}")
    return 0


def knowledge_bump(version: str | None) -> int:
    d = _knowledge_dir()
    current = (d / VERSION_FILE).read_text().strip() if (d / VERSION_FILE).exists() else ""
    if not version:
        today = date.today().isoformat()
        n = int(current.split(".")[-1]) + 1 if current.startswith(today + ".") else 1
        version = f"{today}.{n}"
    (d / VERSION_FILE).write_text(version + "\n")
    (d / CHECKSUM_FILE).write_text(compute_checksum(d) + "\n")
    print(f"{current or '(trống)'} -> {version}")
    return 0


def knowledge_info() -> int:
    k = load_knowledge(_knowledge_dir())
    han = sum(1 for ch in k.text if "一" <= ch <= "鿿")
    # Ước lượng thô: ~1 token cho 1–1.5 chữ Hán, ~3 ký tự Latin (tiếng Việt) cho 1 token.
    est = int(han / 1.2 + (len(k.text) - han) / 3)
    print(json.dumps({"version": k.version, "files": k.files, "chars": len(k.text), "han_chars": han,
                      "estimated_tokens": est}, ensure_ascii=False, indent=2))
    return 0


async def _warmup() -> int:
    from app.container import build_container

    c = build_container(get_settings())
    try:
        results = await c.service.warmup()
    finally:
        await c.aclose()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any("error" in r for r in results) else 0


async def _stats(hours: float) -> int:
    from app.usage_log import UsageLog

    log = UsageLog(get_settings().usage_db_path)
    print(json.dumps(await log.stats(time.time() - hours * 3600), ensure_ascii=False, indent=2))
    log.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    kn = sub.add_parser("knowledge").add_subparsers(dest="action", required=True)
    kn.add_parser("check")
    bump = kn.add_parser("bump")
    bump.add_argument("version", nargs="?")
    kn.add_parser("info")
    sub.add_parser("warmup")
    st = sub.add_parser("stats")
    st.add_argument("--hours", type=float, default=24)
    args = parser.parse_args(argv)

    if args.cmd == "knowledge":
        if args.action == "check":
            return knowledge_check()
        if args.action == "bump":
            return knowledge_bump(args.version)
        return knowledge_info()
    if args.cmd == "warmup":
        return asyncio.run(_warmup())
    return asyncio.run(_stats(args.hours))


if __name__ == "__main__":
    raise SystemExit(main())
