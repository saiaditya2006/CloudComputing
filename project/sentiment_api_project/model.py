from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment(text: str):
    result = sentiment_pipeline(text)[0]
    return result['label'], float(result['score'])