FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc g++ libblas-dev liblapack-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Создаём виртуальное окружение
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Загружаем модель spaCy
RUN python -m spacy download ru_core_news_sm

# Копируем всё остальное
COPY . .

# Порт: Render задаст через переменную окружения PORT
EXPOSE 8000

# Команда запуска с динамическим портом
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
