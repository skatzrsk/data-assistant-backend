import pandas as pd
import spacy
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, AutoModel
import pinecone
import numpy as np
import os
import gc
import torch
import psutil  # Для мониторинга памяти

app = FastAPI()
nlp = spacy.load("ru_core_news_sm")
# Ленивая загрузка моделей Transformers
tokenizer = None
model = None

# Инициализация Pinecone
pinecone_api_key = "pcsk_32sHi8_7c9KNPRgvQq54K6ZJVRcz6XxMRui6TMx53ZUwJiT4qzM8x1qWhoV1Vdfo4H2PuT"
pinecone_env = "us-east-1-aws"
index = None

def init_pinecone():
    global index
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        index = pinecone.Index("phrases")
        return index
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
        raise

def load_models():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        print("Models loaded successfully")

# Проверка доступной памяти
def check_memory(threshold_mb=400):
    memory = psutil.virtual_memory()
    available_mb = memory.available / (1024 * 1024)
    return available_mb > threshold_mb

# Эндпоинт для загрузки файла
@app.post("/upload")
async def upload_file(file: UploadFile):
    global tokenizer, model, index
    try:
        if not check_memory():
            return {"status": "error", "message": "Insufficient memory available"}

        if index is None:
            init_pinecone()
        if tokenizer is None or model is None:
            load_models()

        chunksize = 100
        total_vectors = []
        id_counter = 0

        for chunk in pd.read_excel(file.file, chunksize=chunksize):
            phrases = chunk["Фраза"].tolist()
            themes = chunk["Тематика"].tolist()
            lemmatized = [" ".join([token.lemma_ for token in nlp(phrase)]) for phrase in phrases]

            embeddings = []
            for i in range(0, len(lemmatized), 10):
                batch = lemmatized[i:i + 10]
                if not check_memory():
                    return {"status": "error", "message": "Memory limit reached during batch processing"}
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                embeddings.extend(outputs)

            vectors = [(str(id_counter + i), emb.tolist(), {"themes": themes[i]}) for i, emb in enumerate(embeddings)]
            total_vectors.extend(vectors)
            id_counter += len(phrases)

            del chunk, phrases, themes, lemmatized, embeddings, inputs, outputs
            gc.collect()

        if total_vectors:
            index.upsert(total_vectors)
        return {"status": "success", "processed": id_counter}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Эндпоинт для обработки запроса
@app.post("/query")
async def process_query(query: str):
    global tokenizer, model, index
    try:
        if not check_memory():
            return {"status": "error", "message": "Insufficient memory available"}

        if index is None:
            init_pinecone()
        if tokenizer is None or model is None:
            load_models()

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
        port = int(os.getenv("PORT", 8000))
        print(f"Starting server on port {port}")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
        raise
