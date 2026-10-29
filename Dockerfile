FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Run as a non-root user (also what Hugging Face Spaces expects).
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

USER user

# Bake the ONNX embedding model into the image so startup doesn't download it.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warm-up'])"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "scripts/serve.py"]
