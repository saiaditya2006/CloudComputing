import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

model = load_model()

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Type a sentence below and see the sentiment:")

user_input = st.text_input("Enter text here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        result = model(user_input)[0]
        st.success(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
