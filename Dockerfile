# Serving image: FastAPI + ONNX Runtime (no PyTorch needed at inference time)
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PAM_MODEL_DIR=/app/models PAM_WEB_DIR=/app/web PAM_DATA_DIR=/tmp/pam-results

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY pam/ pam/
COPY web/ web/
COPY models/ models/

RUN useradd --create-home --uid 10001 pam && chown -R pam /app
USER pam

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/ready').status==200 else 1)"
CMD ["uvicorn", "pam.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
