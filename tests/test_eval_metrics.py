from app.router import Classification
from eval.router_eval import Example, metrics


def test_router_metrics():
    ex = [Example("a", "grammar", 0), Example("b", "lookup", 0),
          Example("c", "off_topic", 0), Example("d", "grammar", 1)]
    preds = [
        Classification("lookup", 0.9, 0.1, "x"),   # câu khó bị đẩy sang small
        Classification("lookup", 0.9, 0.1, "x"),
        Classification("off_topic", 0.9, 0.1, "x"),
        None,                                       # router lỗi -> large
    ]
    m = metrics(ex, preds, 0.6)
    assert m["accuracy"] == 0.5
    assert m["hard_to_small_rate"] == 0.5
    assert m["errors"] == 1 and m["large_rate"] == 0.25
    # Ngưỡng cao hơn confidence: mọi câu vào large, không còn câu khó bị đẩy nhầm
    assert metrics(ex, preds, 0.95)["hard_to_small_rate"] == 0.0
