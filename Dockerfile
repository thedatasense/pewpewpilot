FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DURLAR_READ_ONLY=1 \
    DURLAR_POINT_STRIDE=4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py durlar_loader.py download_durlar.py generate_sample_data.py ./
COPY static/ ./static/
COPY calibration/ ./calibration/

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
