FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ libblas-dev liblapack-dev build-essential
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download ru_core_news_sm
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
