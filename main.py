import pandas as pd
import spacy
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, AutoModel
import pinecone
import numpy as np
import os
import gc
import torch

app = FastAPI()
nlp = spacy.load("ru_core_news_sm")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")

# Функция инициализации Pinecone
def init_pinecone():
    try:
        pinecone.init(
            api_key="pcsk_32sHi8_7c9KNPRgvQq54K6ZJVRcz6XxMRui6TMx53ZUwJiT4qzM8x1qWhoV1Vdfo4H2PuT",
            environment="us-east-1-aws"
        )
        return pinecone.Index("phrases")
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
        raise

index = init_pinecone()

# Эндпоинт для загрузки файла с обработкой 2000 фраз
@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # Чтение файла частями по 250 строк
        chunksize = 250
        total_vectors = []
        id_counter = 0

        for chunk in pd.read_excel(file.file, chunksize=chunksize):
            phrases = chunk["Фраза"].tolist()
            themes = chunk["Тематика"].tolist()
            lemmatized = [" ".join([token.lemma_ for token in nlp(phrase)]) for phrase in phrases]

            # Векторизация батчами по 25
            embeddings = []
            for i in range(0, len(lemmatized), 25):
                batch = lemmatized[i:i + 25]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                embeddings.extend(outputs)

            # Подготовка векторов для Pinecone
            vectors = [(str(id_counter + i), emb.tolist(), {"themes": themes[i]}) for i, emb in enumerate(embeddings)]
            total_vectors.extend(vectors)
            id_counter += len(phrases)

            # Очистка памяти
            del chunk, phrases, themes, lemmatized, embeddings, inputs, outputs
            gc.collect()

        # Индексация всех векторов
        if total_vectors:
            index.upsert(total_vectors)
        return {"status": "success", "processed": id_counter}
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
    try:
        port = int(os.getenv("PORT", 8000))  # Динамический порт от Render
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
        raise
