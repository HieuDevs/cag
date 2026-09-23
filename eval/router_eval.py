"""Eval router (docs/routing.md mục 6).

    python -m eval.router_eval                     # chạy mọi router đã cấu hình
    python -m eval.router_eval --backend embedding
    python -m eval.router_eval --data eval/datasets/router.jsonl --json out.json

Chỉ số quan trọng nhất: **tỉ lệ câu khó bị đẩy nhầm sang `small`**, mục tiêu dưới 3%.
Bảng quét ngưỡng giúp chọn ngưỡng confidence đạt mục tiêu đó với tỉ lệ `large` thấp nhất.
"""

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.embeddings import Embedder
from app.llm.openrouter import OpenRouterClient
from app.router import INTENTS, ROUTES, Classification, EmbeddingClassifier, IntentRouter, JevClassifier

DEFAULT_DATA = Path(__file__).parent / "datasets" / "router.jsonl"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Example:
    text: str
    intent: str
    needs_context: int


def true_tier(intent: str) -> str:
    return "canned" if intent == "off_topic" else ROUTES[intent][0]


def metrics(examples: list[Example], preds: list[Classification | None], threshold: float) -> dict:
    router = IntentRouter([])
    n = len(examples)
    correct = hard_to_small = large = canned_wrong = errors = ctx_correct = 0
    hard_total = sum(1 for e in examples if true_tier(e.intent) == "large")
    for e, p in zip(examples, preds, strict=True):
        if p is None:  # router lỗi -> hệ thống gán large
            errors += 1
            large += 1
            continue
        route = router.decide(p, threshold)
        correct += p.intent == e.intent
        ctx_correct += (p.needs_context >= 0.5) == bool(e.needs_context)
        large += route.tier == "large"
        if true_tier(e.intent) == "large" and route.tier == "small":
            hard_to_small += 1
        if route.tier == "canned" and e.intent != "off_topic":
            canned_wrong += 1
    return {
        "threshold": threshold,
        "accuracy": correct / n,
        "needs_context_accuracy": ctx_correct / n,
        "hard_to_small_rate": hard_to_small / hard_total if hard_total else 0.0,
        "large_rate": large / n,
        "wrongly_canned": canned_wrong,
        "errors": errors,
    }


def confusion(examples: list[Example], preds: list[Classification | None]) -> dict[str, Counter]:
    table: dict[str, Counter] = {i: Counter() for i in INTENTS}
    for e, p in zip(examples, preds, strict=True):
        table[e.intent][p.intent if p else "ERROR"] += 1
    return table


async def run_classifier(clf, examples: list[Example], concurrency: int = 8) -> list[Classification | None]:
    sem = asyncio.Semaphore(concurrency)

    async def one(e: Example) -> Classification | None:
        async with sem:
            try:
                return await clf.classify(e.text, None)
            except Exception as ex:  # noqa: BLE001
                print(f"  lỗi với {e.text!r}: {ex!r}")
                return None

    return await asyncio.gather(*(one(e) for e in examples))


def print_report(name: str, examples, preds, default_threshold: float) -> dict:
    print(f"\n=== {name} ({len(examples)} câu) ===")
    rows = [metrics(examples, preds, t) for t in sorted({*THRESHOLDS, default_threshold})]
    print(f"{'ngưỡng':>7} {'đúng intent':>12} {'khó→small':>10} {'tỉ lệ large':>12} {'canned sai':>11} {'lỗi':>5}")
    for r in rows:
        mark = " ←" if r["threshold"] == default_threshold else ""
        print(f"{r['threshold']:>7.2f} {r['accuracy']:>12.1%} {r['hard_to_small_rate']:>10.1%} "
              f"{r['large_rate']:>12.1%} {r['wrongly_canned']:>11} {r['errors']:>5}{mark}")
    ok = [r for r in rows if r["hard_to_small_rate"] < 0.03]
    if ok:
        best = min(ok, key=lambda r: r["large_rate"])
        print(f"Ngưỡng đề xuất: {best['threshold']} (khó→small {best['hard_to_small_rate']:.1%}, "
              f"large {best['large_rate']:.1%})")
    print(f"needs_context đúng: {rows[0]['needs_context_accuracy']:.1%}")
    print("Ma trận nhầm lẫn (hàng = nhãn đúng):")
    table = confusion(examples, preds)
    cols = [*INTENTS, "ERROR"]
    print(" " * 11 + "".join(f"{c[:9]:>10}" for c in cols))
    for intent in INTENTS:
        print(f"{intent:>11}" + "".join(f"{table[intent][c]:>10}" for c in cols))
    return {"name": name, "sweep": rows}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backend", choices=["jev", "embedding", "all"], default="all")
    parser.add_argument("--json", type=Path, help="ghi kết quả ra file JSON")
    args = parser.parse_args()

    settings = get_settings()
    examples = [Example(**json.loads(line)) for line in args.data.read_text("utf-8").splitlines() if line.strip()]
    openrouter = OpenRouterClient(settings)
    classifiers = []
    if not settings.openrouter_api_key:
        raise SystemExit("Cần OPENROUTER_API_KEY (dùng cho cả Jev và embedding)")
    if args.backend in ("jev", "all"):
        # Eval không cần nhanh: nới timeout để đo chất lượng, không đo độ trễ.
        settings.jev_timeout_seconds = 10
        classifiers.append((JevClassifier(settings, openrouter), settings.jev_confidence_threshold))
    if args.backend in ("embedding", "all"):
        embedder = Embedder(openrouter, settings.embedding_model, timeout=10)
        classifiers.append((EmbeddingClassifier(settings, embedder), settings.embedding_confidence_threshold))

    reports = []
    for clf, threshold in classifiers:
        preds = await run_classifier(clf, examples)
        reports.append(print_report(clf.name, examples, preds, threshold))
    await openrouter.aclose()
    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
