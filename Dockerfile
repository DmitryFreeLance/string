# Базовый образ
FROM python:3.11-slim

# Системные либы (для Pillow/Numpy/Skimage)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libjpeg62-turbo-dev zlib1g-dev \
   && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Рабочая директория
WORKDIR /app

# Устанавливаем Python-зависимости
COPY requirements.txt /app/
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . /app

# Нерутовый пользователь и права
RUN useradd -ms /bin/bash appuser \
 && mkdir -p /app/output \
 && chown -R appuser:appuser /app
USER appuser

# Важно: BOT_TOKEN прокинем снаружи (через Compose/`-e`)
ENV TZ=UTC

# Старт (long-polling; портов открывать не надо)
CMD ["python", "bot.py"]