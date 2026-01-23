# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
from transformers import pipeline
from stqdm import stqdm
import plotly.express as px

# ---------------------------
# 1️⃣ Pipelines
# ---------------------------
# Sentiment
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1
)
label_map = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}

# Emotion
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1
)

# Emoji mapping for chart
emoji_map = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😱",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐"
}

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.title("🍔 Restaurant Review Sentiment Dashboard")
st.write("Analyze sentiment and emotion of reviews, and compare with rating.")

# ---------------------------
# 2a️⃣ Single Review Input
# ---------------------------
st.subheader("Single Review Analysis")

user_review = st.text_area("Enter your review:")

# Star rating input using slider
user_rating = st.slider("Select rating (⭐):", 1, 5, 5, 1)

if st.button("Analyze Review"):
    if user_review.strip():
        # --- Sentiment ---
        sentiment_result = sentiment_pipeline(user_review)[0]
        sentiment_label = label_map.get(sentiment_result['label'], sentiment_result['label'])
        sentiment_score = sentiment_result['score']

        # --- Emotion ---
        emotion_results = emotion_pipeline(user_review)[0]
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_results}

        # --- Rating → Sentiment mapping ---
        def rating_to_sentiment(r):
            if r >= 4:
                return "positive"
            elif r == 3:
                return "neutral"
            else:
                return "negative"

        rating_sentiment = rating_to_sentiment(user_rating)

        # --- Display Sentiment ---
        st.subheader("Sentiment Analysis")
        st.write(f"**Predicted Sentiment:** {sentiment_label} (Confidence: {sentiment_score:.2f})")
        st.write(f"**Rating Sentiment:** {rating_sentiment} (⭐ {user_rating})")

        # --- Emotion Chart ---
        df_emotion = pd.DataFrame({
            "Emotion": [f"{emoji_map.get(k, '')} {k.capitalize()}" for k in emotion_dict.keys()],
            "Score": list(emotion_dict.values())
        })
        fig = px.bar(
            df_emotion,
            x="Emotion",
            y="Score",
            text="Score",
            color="Score",
            color_continuous_scale="viridis"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.subheader("Emotion Analysis")
        st.plotly_chart(fig)

        # --- Mismatch Check ---
        st.subheader("Sentiment vs Rating Check")
        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Mismatch detected!")
            st.dataframe(pd.DataFrame([{
                "Review": user_review,
                "Rating": user_rating,
                "Rating Sentiment": rating_sentiment,
                "Predicted Sentiment": sentiment_label,
                "Sentiment Confidence": sentiment_score
            }]))
        else:
            st.success("✅ No mismatch detected.")

# ---------------------------
# 2b️⃣ Batch CSV Upload
# ---------------------------
st.subheader("Batch Review Analysis (CSV)")
uploaded_file = st.file_uploader("Upload CSV with 'review' and 'rating' columns:", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"CSV loaded successfully ({len(df)} rows).")
    st.dataframe(df.head())

    # Clean rating column
    df['rating'] = (
        df['rating']
        .astype(str)
        .str.extract(r'(\d)')  # extract 1–5 from "5 stars"
        .astype(float)
    )

    # Rating → Sentiment mapping
    def rating_to_sentiment(x):
        if pd.isna(x):
            return None
        if x >= 4:
            return "positive"
        elif x == 3:
            return "neutral"
        else:
            return "negative"

    df['rating_sentiment'] = df['rating'].apply(rating_to_sentiment)

    if st.button("Analyze CSV Reviews"):
        reviews = df['review'].astype(str).tolist()
        all_results = []

        # Process reviews with progress bar
        for review in stqdm(reviews, desc="Processing reviews"):
            result = sentiment_pipeline(review)[0]
            all_results.append(result)

        df['predicted_sentiment'] = [label_map.get(r['label'], r['label']) for r in all_results]
        df['sentiment_confidence'] = [r['score'] for r in all_results]

        # Display mismatches
        mismatches = df[df['predicted_sentiment'] != df['rating_sentiment']]
        if len(mismatches) > 0:
            st.warning(f"⚠️ Found {len(mismatches)} mismatched reviews")
            st.dataframe(mismatches[['review', 'rating', 'rating_sentiment', 'predicted_sentiment', 'sentiment_confidence']])
        else:
            st.success("✅ No mismatches found in the dataset!")
