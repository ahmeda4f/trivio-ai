from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# ===== CONFIGURATION =====

MODEL_ID = "Ahmed-Ashraf-00/egyptian-content-filter-model" 

app = FastAPI(title="Egyptian Arabic Content Filter API")

# ===== Load Models =====
classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    truncation=True,    
    max_length=512
)

sentiment_analyzer = pipeline(
    "text-classification",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
    truncation=True,    
    max_length=512
)

class TextRequest(BaseModel):
    text: str

# ===== Logic =====
def smart_filter_normalized(text, offensive_threshold=0.99):
    hate_result = classifier(text)[0]
    hate_label = hate_result['label']
    hate_score = hate_result['score']

    sentiment_result = sentiment_analyzer(text)[0]
    sentiment_label = sentiment_result['label']

    if hate_score < offensive_threshold:
        return f"✅ Safe (Low Confidence: {hate_score:.2f})"

    if hate_label in ["Racism", "Religious Discrimination", "Sexism"]:
        return f"❌ BLOCKED (Policy Violation: {hate_label})"

    if sentiment_label == "neutral":
        return f"✅ Safe"

    if hate_label == "Offensive":
        if sentiment_label == "positive":
            return "✅ Safe (Slang/Praise Context)"
        if sentiment_label == "negative":
            return "⚠️ Flagged for Masking (Aggressive Critic)"
        return "✅ Safe"

    return "✅ Safe"

@app.post("/predict")
def predict(request: TextRequest):
    result = smart_filter_normalized(request.text)
    return {"result": result}