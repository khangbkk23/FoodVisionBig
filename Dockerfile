FROM python:3.11.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN echo "Start installing dependencies..."
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user

COPY --chown=user:user . .

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

EXPOSE 7860
RUN echo "Running collectstatic..."
WORKDIR /app/app
RUN mkdir -p media && \
    python manage.py collectstatic --noinput

CMD ["sh", "-c", "python manage.py migrate && gunicorn api.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --worker-class gthread --timeout 120"]