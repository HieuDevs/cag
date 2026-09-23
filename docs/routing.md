# Định tuyến bằng Jev

## 1. Jev là gì

[Jev](https://docs.typesafe.ai/) (TypeSafe) là một "System One model". Jev **không sinh văn bản** mà trả về quyết định có cấu trúc (`Choice`, `Score`, `Noul`), kèm xác suất cho từng lựa chọn và `confidence`. Nhiều câu hỏi được xử lý song song trong một lần gọi.

**Jev gọi qua OpenRouter**, dùng chính `OPENROUTER_API_KEY` và tính tiền vào tài khoản OpenRouter. Không cần key TypeSafe riêng, cũng không cần cài `typesafe-sdk`. Xem [hướng dẫn Jev của OpenRouter](https://openrouter.ai/docs/guides/community/jev).

| Thông số (tra ngày 2026-09-23) | Giá trị |
|---|---|
| Model trên OpenRouter | `typesafe/jev-1.13` (alias `~typesafe/jev-latest`, không dùng vì tự đổi version) |
| Endpoint | `POST https://openrouter.ai/api/v1/systemone` (System One API, cùng định dạng với API của TypeSafe). OpenRouter còn có Decisions API ở `POST /api/alpha/decisions`, đang ở bản alpha |
| Giá | $0.042 / 1M token input, output miễn phí |
| Độ trễ | 70–500ms (theo TypeSafe; đi qua OpenRouter có thể thêm một chút) |
| Context | 32K token trên OpenRouter |
| Rate limit | Theo tài khoản OpenRouter và giới hạn của TypeSafe (**đang được điều chỉnh liên tục**) |
| Ngôn ngữ | Tiếng Anh tốt nhất. Ngôn ngữ khác, kể cả chữ Hán, "xử lý được nhưng kém hơn" |
| Dữ liệu | Theo chính sách của TypeSafe với vai trò nhà cung cấp trên OpenRouter |

**Chi phí ước lượng:** khoảng 400 token mỗi request × 1M request = 400M token, tức **khoảng $17/tháng**.

## 2. Các câu hỏi gửi Jev mỗi request

| ID | Loại | Mục đích |
|---|---|---|
| `intent` | Choice | Xác định loại câu hỏi để chọn tầng model, độ dài câu trả lời, có cache không |
| `needs_context` | Noul | Câu hỏi có nhắc tới lượt trước không ("từ đó", "câu trên"). Nếu có thì không dùng cache câu trả lời |

Mô tả các lựa chọn (`criteria`) viết bằng **tiếng Anh**, vì đây là ngôn ngữ Jev làm tốt nhất. Câu hỏi của user giữ nguyên ngôn ngữ gốc trong `state`.

## 3. Bảng định tuyến

| Intent | Mô tả | Tầng | Cache câu trả lời | Output tối đa |
|---|---|---|---|---|
| `lookup` | Nghĩa, pinyin, thanh điệu, thứ tự nét, âm Hán Việt của một từ hoặc cụm ngắn | small | ✅ | ~300 token |
| `translate` | Dịch một câu ngắn Việt ↔ Trung | small | ✅ | ~300 token |
| `grammar` | Giải thích hoặc so sánh ngữ pháp, trợ từ, lượng từ, mẫu câu | large | ✅ | ~1.000 token |
| `culture` | Lịch sử, văn hóa, điển tích thành ngữ, cổ văn | large | ✅ | ~1.000 token |
| `correction` | Chữa câu hoặc đoạn văn user tự viết | large | ❌ (mang tính cá nhân) | ~1.200 token |
| `off_topic` | Không liên quan tới học tiếng Trung | câu mẫu | — | — |
| confidence < ngưỡng | Router không chắc chắn | large | ❌ | ~1.000 token |
| Jev lỗi hoặc timeout | Chuyển sang router embedding (`bge-m3` + câu mẫu có nhãn) | theo kết quả router embedding | theo intent | theo intent |
| Cả hai router lỗi | | large | ❌ | ~1.000 token |

**Nguyên tắc an toàn:** khi không chắc chắn thì luôn chọn `large`. Tốn thêm chút tiền vẫn tốt hơn trả lời sai học thuật.

**Quyền dùng model đắt kiểm tra bằng code** theo gói của user, không để Jev quyết định (làm cùng quota, sau bản test). Nội dung user nhập có thể cố tình lái kết quả của router.

## 4. Code mẫu

Bản rút gọn để minh họa. Code thật ở [app/router.py](../app/router.py) (`JevClassifier`, `IntentRouter`) và [app/llm/openrouter.py](../app/llm/openrouter.py) (`OpenRouterClient.system_one`), có thêm router embedding dự phòng.

```python
import asyncio
import os
from dataclasses import dataclass

import httpx

JEV_URL = "https://openrouter.ai/api/v1/systemone"
JEV_MODEL = "typesafe/jev-1.13"  # ghim version: ngưỡng confidence tune theo version

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the learner asking the Chinese-learning assistant to do?",
        "criteria": {
            "lookup": "Meaning, pinyin, tones, stroke order or Sino-Vietnamese reading of one word or short phrase",
            "translate": "Translate one short sentence between Vietnamese and Chinese",
            "grammar": "Explain or compare grammar points, particles, measure words or sentence patterns",
            "correction": "Check or correct a sentence or paragraph the learner wrote themselves",
            "culture": "Chinese history, culture, idiom origins or classical Chinese",
            "off_topic": "Not related to learning Chinese at all",
        },
    },
    "needs_context": {
        "type": "noul",
        "instructions": "Does the question refer to something earlier in the conversation, "
                        "such as 'that word' or 'the sentence above'?",
    },
}

ROUTES = {  # intent -> (tầng, có được cache câu trả lời không)
    "lookup": ("small", True),
    "translate": ("small", True),
    "grammar": ("large", True),
    "culture": ("large", True),
    "correction": ("large", False),
}

CONFIDENCE_THRESHOLD = 0.6  # giá trị tạm, chốt bằng eval (mục 6)

http = httpx.AsyncClient(headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})

@dataclass
class Route:
    tier: str        # "small" | "large" | "canned"
    cacheable: bool
    reason: str

async def route(question: str) -> Route:
    try:
        resp = await asyncio.wait_for(
            http.post(JEV_URL, json={"model": JEV_MODEL, "state": {"question": question}, "questions": QUESTIONS}),
            timeout=0.8,
        )
        resp.raise_for_status()
        answers = resp.json()["answers"]
    except Exception:
        return Route("large", False, "jev_unavailable")  # code thật: chuyển sang router embedding trước

    intent = answers["intent"]  # {"type": "choice", "choice": ..., "confidence": ..., "probabilities": {...}}
    if intent["confidence"] < CONFIDENCE_THRESHOLD:
        return Route("large", False, "low_confidence")
    if intent["choice"] == "off_topic":
        return Route("canned", False, "off_topic")

    tier, cacheable = ROUTES[intent["choice"]]
    cacheable = cacheable and answers["needs_context"]["noul"] < 0.5  # {"type": "noul", "noul": 0..1}
    return Route(tier, cacheable, intent["choice"])
```

## 5. Rủi ro và cách xử lý

Các rủi ro dưới đây lấy từ [trang giới hạn của jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) và [trang Models](https://docs.typesafe.ai/models.md).

| Rủi ro | Cách xử lý |
|---|---|
| Tiếng Việt và chữ Hán không phải ngôn ngữ mạnh nhất của Jev | `criteria` viết tiếng Anh. Eval trên câu hỏi thật. Giữ bộ phân loại embedding làm dự phòng |
| Jev hiểu theo nghĩa đen | Mô tả từng lựa chọn cụ thể, dùng dạng object (phạm vi bao gồm, không bao gồm, ví dụ) khi cần |
| Nội dung user cố tình lái kết quả | Quyền truy cập theo gói kiểm tra trong code. Theo dõi phân bố intent bất thường |
| Rate limit và giá có thể thay đổi, vì đây là startup mới | Timeout 800ms, lỗi thì gán `large`. Dự phòng bằng embedding. Theo dõi tỉ lệ lỗi |
| Alias `~typesafe/jev-latest` tự đổi version | Ghim `typesafe/jev-1.13` (`JEV_MODEL`). Khi nâng version phải chạy lại eval và tune lại ngưỡng |
| Không nên suy luận xác suất giữa Noul và Choice | Mỗi ngưỡng tune riêng cho từng câu hỏi |

## 6. Eval cho router

1. Gom **khoảng 300 câu hỏi thật**, trộn câu tiếng Việt, câu tiếng Trung, câu viết tắt, không dấu, sai chính tả. Gán nhãn `intent` và `needs_context` bằng tay.
2. Chạy 2 router trên cùng bộ dữ liệu:
   - Jev.
   - `bge-m3` + logistic regression, train trên một phần dữ liệu có nhãn và test trên phần còn lại.
3. Đo:
   - Độ chính xác theo từng intent, và ma trận nhầm lẫn.
   - **Tỉ lệ câu khó bị đẩy nhầm sang `small`.** Đây là chỉ số quan trọng nhất, mục tiêu **dưới 3%**.
   - Đường calibration của `confidence`. Chọn `CONFIDENCE_THRESHOLD` đạt mục tiêu trên với tỉ lệ `large` thấp nhất.
4. Nếu bộ embedding ngang ngửa Jev thì cân nhắc dùng embedding làm router chính, vì không tốn chi phí mỗi request và không phụ thuộc bên thứ ba.
