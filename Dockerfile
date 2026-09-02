FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure sample data + candidate model exist for local/dev runs
RUN python scripts/generate_sample_data.py && \
    python scripts/validate_and_gate.py || true

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
