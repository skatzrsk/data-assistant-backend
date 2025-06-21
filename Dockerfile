FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Создание виртуального окружения
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Установка модели SpaCy
RUN python -m spacy download ru_core_news_sm

# Копирование кода приложения
COPY . .

# Установка прав доступа
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Открытие порта
EXPOSE 8000

# Запуск приложения с динамическим портом
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
