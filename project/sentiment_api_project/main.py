from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sentiment

app = FastAPI(title="Cloud Based Sentiment Analysis API")

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: SentimentRequest):
    sentiment, confidence = predict_sentiment(request.text)
    return {"sentiment": sentiment, "confidence": confidence}