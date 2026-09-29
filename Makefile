PYTHON ?= .venv/bin/python
PORT ?= 8000
# Mạng có proxy chặn và kiểm tra HTTPS (ví dụ Fortinet): ghép CA của proxy với certifi vào file này,
# `make run` tự đặt SSL_CERT_FILE. Không có file thì chạy bình thường.
CA_BUNDLE := data/ca-bundle.pem

.PHONY: run
run:
	$(if $(wildcard $(CA_BUNDLE)),SSL_CERT_FILE=$(CA_BUNDLE)) $(PYTHON) -m uvicorn app.main:app --reload --port $(PORT)
