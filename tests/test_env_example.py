"""`.env.example` phải liệt kê đủ mọi biến cấu hình, và không biến nào để trống."""

import pytest
from pydantic import ValidationError

from app.config import PLACEHOLDER, ROOT_DIR, Settings

ENV_EXAMPLE = ROOT_DIR / ".env.example"
REQUIRED = {"OPENROUTER_API_KEY"}


def parse_env(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        assert sep, f"Dòng không đúng dạng KEY=VALUE: {line!r}"
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        assert key not in values, f"{key} bị khai báo hai lần"
        values[key] = value
    return values


@pytest.fixture
def example() -> dict[str, str]:
    return parse_env(ENV_EXAMPLE.read_text("utf-8"))


def test_env_example_lists_every_setting(example):
    fields = {name.upper() for name in Settings.model_fields}
    assert set(example) - fields == set(), "Biến trong .env.example không có trong Settings"
    assert fields - set(example) == set(), "Thêm biến vào Settings thì phải thêm vào .env.example"


def test_env_example_has_no_empty_value(example):
    empty = [k for k, v in example.items() if not v.strip()]
    assert empty == []


def test_required_secrets_are_placeholders(example):
    assert {k for k, v in example.items() if v == PLACEHOLDER} == REQUIRED


def test_env_example_defaults_match_config(example, monkeypatch):
    """Giá trị trong .env.example (trừ key bí mật) phải đúng bằng mặc định trong app/config.py."""
    for k, v in example.items():
        monkeypatch.setenv(k, "real-secret" if k in REQUIRED else v)
    from_file = Settings(_env_file=None)
    for k in example:
        monkeypatch.delenv(k)
    defaults = Settings(_env_file=None, openrouter_api_key="real-secret")
    for name in Settings.model_fields:
        a, b = getattr(from_file, name), getattr(defaults, name)
        if name in ("usage_db_path", "knowledge_dir"):  # file ghi đường dẫn tương đối
            a, b = (ROOT_DIR / a).resolve(), b.resolve()
        assert a == b, f"{name.upper()}: .env.example={a!r}, config.py={b!r}"


@pytest.mark.parametrize("value", ["", "   ", PLACEHOLDER])
def test_required_secret_rejected(value):
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(_env_file=None, openrouter_api_key=value)


def test_missing_required_secret_fails(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("field,value", [
    ("openrouter_data_collection", "maybe"),
    ("router_backend", "gpt"),
    ("redis_url", "localhost:6379"),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openrouter_api_key="k", **{field: value})


def test_env_example_loads_through_dotenv_and_only_secrets_fail(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    with pytest.raises(ValidationError) as ei:
        Settings(_env_file=ENV_EXAMPLE)
    assert {e["loc"][0] for e in ei.value.errors()} == {"openrouter_api_key"}


def test_quoted_env_vars_are_unquoted(monkeypatch):
    """`docker run --env-file` truyền nguyên dấu nháy vào biến môi trường."""
    monkeypatch.setenv("TIER_SMALL", """'[{"model":"a/b","providers":["x"]}]'""")
    monkeypatch.setenv("OPENROUTER_APP_NAME", '"CAG Test"')
    s = Settings(_env_file=None, openrouter_api_key="k")
    assert s.tier_small[0].model == "a/b" and s.openrouter_app_name == "CAG Test"
