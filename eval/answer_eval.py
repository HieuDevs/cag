"""Eval câu trả lời (docs/roadmap.md, tiêu chí chấm).

    python -m eval.answer_eval                          # chạy bộ câu hỏi qua toàn bộ pipeline
    python -m eval.answer_eval --judge anthropic/claude-sonnet-5
    python -m eval.answer_eval --only grammar --out tmp/answers.jsonl

Gọi OpenRouter thật (tốn tiền). Mỗi câu dùng user riêng và store trong bộ nhớ, nên cache câu trả lời
không ảnh hưởng kết quả. Kết quả ghi ra JSONL để người chấm xem lại.

Tự động chấm: có đủ cụm bắt buộc (`must_include`), độ dài so với giới hạn của intent, chi phí, độ trễ.
Tùy chọn `--judge`: một model mạnh chấm 1–5 theo các tiêu chí học thuật và tiếng Việt.
"""

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

from app.config import ModelTarget, get_settings
from app.container import build_container
from app.llm.types import ChatRequest
from app.router import MAX_OUTPUT_TOKENS
from app.service import ChatInput
from app.store import MemoryStore

DEFAULT_DATA = Path(__file__).parent / "datasets" / "answers.jsonl"

JUDGE_PROMPT = """Bạn là giáo viên tiếng Trung chấm câu trả lời của một trợ lý dạy tiếng Trung cho người Việt.

Câu hỏi của người học (trình độ {level}):
{question}

Câu trả lời cần chấm:
{answer}

Chấm từ 1 (rất kém) tới 5 (xuất sắc) theo từng tiêu chí:
- academic: đúng học thuật (pinyin, thanh điệu, nghĩa, ngữ pháp, ví dụ đúng)
- vietnamese: tiếng Việt tự nhiên, dễ hiểu
- no_mixing: không chen chữ Trung hay tiếng Anh sai chỗ
- level_fit: phù hợp trình độ, độ dài hợp lý

Chỉ trả về JSON:
{{"academic": n, "vietnamese": n, "no_mixing": n, "level_fit": n, "errors": ["lỗi sai cụ thể nếu có"]}}"""


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


async def judge(container, model: str, item: dict, answer: str) -> dict:
    level = item.get("level") or "không rõ"
    prompt = JUDGE_PROMPT.format(level=level, question=item["question"], answer=answer)
    req = ChatRequest(messages=[{"role": "user", "content": prompt}], max_tokens=600, temperature=0)
    text = ""
    async for ev in container.openrouter.stream_chat(ModelTarget(model=model), req):
        text += ev.text
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)) if match else {"raw": text}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--only", help="chỉ chạy một intent")
    parser.add_argument("--judge", help="model chấm, ví dụ anthropic/claude-sonnet-5")
    parser.add_argument("--out", type=Path, default=Path("tmp/answer_eval.jsonl"))
    args = parser.parse_args()

    settings = get_settings()
    settings.usage_db_path = args.out.with_suffix(".sqlite3")
    items = [json.loads(line) for line in args.data.read_text("utf-8").splitlines() if line.strip()]
    if args.only:
        items = [i for i in items if i["intent"] == args.only]
    container = build_container(settings, store=MemoryStore())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for n, item in enumerate(items):
            out = {"text": "", "meta": {}, "done": {}, "error": None}
            async for ev in container.service.chat(ChatInput(user_id=f"eval-{n}", message=item["question"],
                                                             level=item.get("level"))):
                if ev["event"] == "delta":
                    out["text"] += ev["data"]["text"]
                else:
                    out[ev["event"]] = ev["data"]
            answer = out["text"]
            missing = [s for s in item.get("must_include", []) if norm(s) not in norm(answer)]
            usage = (out["done"] or {}).get("usage") or {}
            limit = MAX_OUTPUT_TOKENS.get(item["intent"], MAX_OUTPUT_TOKENS[None])
            row = {
                "id": item["id"], "expected_intent": item["intent"], "routed_intent": out["meta"].get("intent"),
                "tier": out["meta"].get("tier"), "model": (out["done"] or {}).get("model"),
                "missing": missing, "output_tokens": usage.get("output_tokens"),
                "over_length": bool(usage.get("output_tokens") and usage["output_tokens"] > limit),
                "cost_usd": (out["done"] or {}).get("cost_usd"), "ttft_ms": (out["done"] or {}).get("ttft_ms"),
                "cache_read_ratio": usage.get("cache_read_ratio"), "error": out["error"], "answer": answer,
            }
            if args.judge and answer and not out["error"] and row["tier"] != "canned":
                row["judge"] = await judge(container, args.judge, item, answer)
            results.append(row)
            status = "LỖI" if out["error"] else ("OK" if not missing else f"thiếu {missing}")
            print(f"[{row['id']}] {row['routed_intent']}/{row['tier']} {row['model']} "
                  f"${row['cost_usd'] or 0:.5f} {row['ttft_ms']}ms  {status}")
    finally:
        await container.aclose()

    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in results if not r["error"]]
    print(f"\n{len(results)} câu, {len(results) - len(ok)} lỗi")
    print(f"Đủ cụm bắt buộc: {sum(not r['missing'] for r in ok)}/{len(ok)}")
    print(f"Đúng intent: {sum(r['routed_intent'] == r['expected_intent'] for r in results)}/{len(results)}")
    costs = [r["cost_usd"] for r in ok if r["cost_usd"]]
    if costs:
        print(f"Chi phí trung bình: ${statistics.mean(costs):.5f}/câu")
    ratios = [r["cache_read_ratio"] for r in ok if r["cache_read_ratio"] is not None]
    if ratios:
        print(f"Tỉ lệ đọc cache trung bình: {statistics.mean(ratios):.1%} (câu đầu tiên thường chưa có cache)")
    judged = [r["judge"] for r in results if isinstance(r.get("judge"), dict) and "academic" in r["judge"]]
    for key in ("academic", "vietnamese", "no_mixing", "level_fit"):
        if judged:
            print(f"Judge {key}: {statistics.mean(j[key] for j in judged):.2f}/5")
    print(f"Chi tiết: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
