import pandas as pd
import spacy
from fastapi import FastAPI, UploadFile
from transformers import AutoTokenizer, AutoModel
import pinecone
import numpy as np

app = FastAPI()
nlp = spacy.load("ru_core_news_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")
pinecone.init(api_key="pcsk_32sHi8_7c9KNPRgvQq54K6ZJVRcz6XxMRui6TMx53ZUwJiT4qzM8x1qWhoV1Vdfo4H2PuT", environment="us-east-1-aws")
index = pinecone.Index("phrases")

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        df = pd.read_excel(file.file)
        phrases = df["Фраза"].tolist()
        themes = df["Тематика"].tolist()
        lemmatized = [" ".join([token.lemma_ for token in nlp(phrase)]) for phrase in phrases]
        embeddings = [
            model(**tokenizer(phrase, return_tensors="pt")).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            for phrase in lemmatized
        ]
        index.upsert(vectors=[(str(i), embeddings[i], {"themes": themes[i]}) for i in range(len(phrases))])
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/query")
async def process_query(query: str):
    try:
        lemmatized_query = " ".join([token.lemma_ for token in nlp(query)])
        query_embedding = model(**tokenizer(lemmatized_query, return_tensors="pt")).last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        results = index.query(query_embedding, top_k=1)
        matched_themes = results["matches"][0]["metadata"]["themes"]
        return {"themes": matched_themes}
    except Exception as e:
        return {"themes": f"Ошибка: {str(e)}"}