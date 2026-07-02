FROM python:3.11-slim

# Install system dependencies required by dlib wheel and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port Render provides via $PORT
ENV PORT=8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
