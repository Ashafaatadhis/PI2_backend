
FROM python:3.11.11-slim

WORKDIR /app

COPY requirements.txt .
COPY model_tf.keras .
COPY scaler.joblib .  
COPY app.py .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]