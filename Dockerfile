FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libjpeg62-turbo-dev zlib1g-dev \
      libfreetype6 libpng16-16 \
      fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

COPY . /app

# (если остаёшься под обычным пользователем внутри образа)
RUN useradd -ms /bin/bash appuser \
 && mkdir -p /app/output \
 && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 TZ=UTC
CMD ["python", "bot.py"]