FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY pyproject.toml README.md ./

RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch==2.6.0

COPY src ./src

RUN python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "reliability_platform.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
