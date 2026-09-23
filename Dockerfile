FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    KNOWLEDGE_DIR=/srv/knowledge USAGE_DB_PATH=/srv/data/usage.sqlite3

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir -e .

COPY knowledge ./knowledge

EXPOSE 8000
# 1 worker cho mỗi container: chỉ mục cache gần giống nằm trong bộ nhớ tiến trình.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
