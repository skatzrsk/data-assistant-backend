import pandas as pd
import spacy
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, AutoModel
import pinecone
import numpy as np
import os

app = FastAPI()
nlp = spacy.load("ru_core_news_sm")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

# Функция инициализации Pinecone
def init_pinecone():
    pinecone.init(
        api_key="pcsk_32sHi8_7c9KNPRgvQq54K6ZJVRcz6XxMRui6TMx53ZUwJiT4qzM8x1qWhoV1Vdfo4H2PuT",
        environment="us-east-1-aws"
    )
    return pinecone.Index("phrases")

index = init_pinecone()

# Эндпоинт для загрузки файла с оптимизацией
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # Ограничение до 50 строк для теста
        df = pd.read_excel(file.file)
        df = df.head(50)
        phrases = df["Фраза"].tolist()
        themes = df["Тематика"].tolist()

        # Лемматизация и векторизация по батчам
        lemmatized = []
        for i in range(0, len(phrases), 10):  # Батчи по 10
            batch = phrases[i:i + 10]
            lemmatized.extend([" ".join([token.lemma_ for token in nlp(phrase)]) for phrase in batch])

        embeddings = []
        for i in range(0, len(lemmatized), 10):  # Батчи по 10
            batch = lemmatized[i:i + 10]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():  # Экономим память
                outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            embeddings.extend(outputs)

        # Индексация в Pinecone
        vectors = [(str(i), embeddings[i], {"themes": themes[i]}) for i in range(len(phrases))]
        index.upsert(vectors)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Эндпоинт для обработки запроса
@app.post("/query")
async def process_query(query: str):
    try:
        lemmatized_query = " ".join([token.lemma_ for token in nlp(query)])
        inputs = tokenizer(lemmatized_query, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        results = index.query(query_embedding, top_k=1)
        matched_themes = results["matches"][0]["metadata"]["themes"]
        return {"themes": matched_themes}
    except Exception as e:
        return {"themes": f"Ошибка: {str(e)}"}

# Точка входа для запуска сервера
if __name__ == "__main__":
    import uvicorn
    import torch
    port = int(os.getenv("PORT", 8000))  # Динамический порт от Render
    uvicorn.run(app, host="0.0.0.0", port=port)
