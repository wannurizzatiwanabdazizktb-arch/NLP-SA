# streamlit_sentiment_dashboard.py

import os
import streamlit as st
# 🔥 CRITICAL FIX FOR STREAMLIT CLOUD
os.environ["TORCH_DISABLE_SDPA"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from stqdm import stqdm

# ---------------------------
# 1️⃣ MODELS
# ---------------------------

# --- Sentiment pipeline (this one is fine) ---
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1,
    truncation=True,
    max_length=512
)

label_map = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}

# --- Emotion model (DIRECT, NO PIPELINE) ---
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

emotion_tokenizer, emotion_model = load_emotion_model()

def get_emotions(text: str) -> dict:
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]

    labels = emotion_model.config.id2label

    return {
        labels[i].lower(): probs[i].item()
        for i in range(len(labels))
    }

# Emoji mapping
emoji_map = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😱",
    "surprise": "😲",
    "neutral": "😐",
    "disgust": "😒"
}

# ---------------------------
# 2️⃣ STREAMLIT UI
# ---------------------------

st.title("🍔 Restaurant Review Sentiment Dashboard")
st.write("Analyze sentiment and emotion of reviews, and compare with rating.")

# --- Single Review ---
st.subheader("Single Review Analysis")

user_review = st.text_area("Enter your review:")

user_rating = st.radio(
    "Rate the restaurant:",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

if st.button("Analyze Review"):
    if user_review.strip():

        # --- Sentiment ---
        sentiment_result = sentiment_pipeline(user_review)[0]
        sentiment_label = label_map.get(
            sentiment_result["label"],
            sentiment_result["label"]
        )
        sentiment_score = sentiment_result["score"]

        # --- Emotion (FIXED) ---
        emotion_dict = get_emotions(user_review)

        # --- Rating sentiment ---
        rating_sentiment = rating_to_sentiment(user_rating)

        # --- Display ---
        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating Sentiment:** {rating_sentiment}")

        # --- Emotion chart ---
        st.subheader("Emotion Analysis")

        df_emotion = pd.DataFrame({
            "Emotion": [
                f"{emoji_map.get(k, '')} {k.capitalize()}"
                for k in emotion_dict.keys()
            ],
            "Score": [v * 100 for v in emotion_dict.values()]
        }).sort_values("Score", ascending=True)

        fig = px.bar(
            df_emotion,
            x="Score",
            y="Emotion",
            orientation="h",
            text=df_emotion["Score"].round(1).astype(str) + "%",
            color="Score",
            color_continuous_scale="Viridis",
            title="Emotion Confidence (%)"
        )

        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="",
            xaxis_range=[0, 100]
        )

        st.plotly_chart(fig)

        # --- Mismatch check ---
        st.subheader("Sentiment vs Rating Check")

        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Mismatch detected!")
            st.dataframe(pd.DataFrame([{
                "Review": user_review,
                "Rating": "⭐" * user_rating,
                "Rating Sentiment": rating_sentiment,
                "Predicted Sentiment": sentiment_label,
                "Confidence": sentiment_score
            }]))
        else:
            st.success("✅ No mismatch detected.")

# ---------------------------
# 3️⃣ CSV BATCH ANALYSIS
# ---------------------------

st.subheader("Batch Review Analysis (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV with 'review' and 'rating' columns:",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"CSV loaded successfully ({len(df)} rows)")
    st.dataframe(df.head())

    df["rating"] = (
        df["rating"]
        .astype(str)
        .str.extract(r"(\d)")
        .astype(float)
    )

    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    if st.button("Analyze CSV Reviews"):
        reviews = df["review"].astype(str).tolist()
        results = []

        for review in stqdm(reviews, desc="Processing reviews"):
            if not review.strip():
                results.append({"label": "NEU", "score": 0.0})
                continue

            try:
                results.append(sentiment_pipeline(review)[0])
            except Exception:
                results.append({"label": "NEU", "score": 0.0})

        df["predicted_sentiment"] = [
            label_map.get(r["label"], r["label"]) for r in results
        ]
        df["sentiment_confidence"] = [r["score"] for r in results]

        mismatches = df[df["predicted_sentiment"] != df["rating_sentiment"]]

        if len(mismatches):
            st.warning(f"⚠️ Found {len(mismatches)} mismatched reviews")
            st.dataframe(mismatches[
                ["review", "rating", "rating_sentiment",
                 "predicted_sentiment", "sentiment_confidence"]
            ])
        else:
            st.success("✅ No mismatches found!")
