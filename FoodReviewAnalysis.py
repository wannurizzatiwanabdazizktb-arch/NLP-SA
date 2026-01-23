# restaurant_review_dashboard.py

import streamlit as st
import pandas as pd
from transformers import pipeline
from stqdm import stqdm
import torch

# ---------------------------
# 1️⃣ Initialize pipelines
# ---------------------------

# Sentiment pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    truncation=True,
    max_length=512,
    device=-1,  # use CPU (-1), set to 0 for GPU
    batch_size=16
)
label_map = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}

# Emotion pipeline
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1
)

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------

st.title("🍽️ Restaurant Review Analyzer")
st.write("Analyze sentiment and emotion of reviews and compare with rating.")
st.write(torch.cuda.is_available())  # must return True
st.write("success")  # must return True

# --- Single Review Input ---
st.subheader("Single Review Analysis")
user_review = st.text_area("Enter your review:")
user_rating = st.number_input("Enter rating (1-5):", min_value=1, max_value=5, value=5, step=1)

if st.button("Analyze Review"):
    if user_review.strip() != "":
        # Sentiment
        sentiment_result = sentiment_pipeline(user_review)[0]
        sentiment_label = label_map.get(sentiment_result['label'], sentiment_result['label'])
        sentiment_conf = sentiment_result['score']

        # Emotion
        emotion_results = emotion_pipeline(user_review)[0]
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_results}

        # Rating → sentiment
        def rating_to_sentiment(rating):
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral"
            else:
                return "negative"
        rating_sentiment = rating_to_sentiment(user_rating)

        # Display
        st.subheader("Sentiment Analysis")
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_conf:.2f}")
        st.write(f"**Rating Sentiment:** {rating_sentiment}")

        st.subheader("Emotion Analysis")
        for emotion, score in emotion_dict.items():
            st.write(f"{emotion.capitalize()}: {score:.2f}")

        st.subheader("Rating vs Sentiment Check")
        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Mismatch detected!")
            st.dataframe(pd.DataFrame([{
                "Review": user_review,
                "Rating": user_rating,
                "Rating Sentiment": rating_sentiment,
                "Predicted Sentiment": sentiment_label,
                "Sentiment Confidence": sentiment_conf,
                **{f"Emotion {k}": v for k, v in emotion_dict.items()}
            }]))
        else:
            st.success("✅ Rating matches sentiment.")

# --- CSV Upload ---
st.subheader("Batch Review Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with 'review' and 'rating' columns", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"CSV loaded successfully with {len(df)} rows.")
    st.dataframe(df.head())

    # Clean rating
    df['rating'] = (
        df['rating']
        .astype(str)
        .str.extract(r'(\d)')   # extract 1–5
        .astype(float)
    )

    # Rating → sentiment mapping
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
        all_sentiment_results = []
        all_emotion_results = []

        chunk_size = 500
        batch_size = 16

        for i in stqdm(range(0, len(reviews), chunk_size), desc="Processing chunks"):
            batch = reviews[i:i+chunk_size]
            batch_sentiments = sentiment_pipeline(batch, truncation=True, max_length=512, batch_size=batch_size)
            all_sentiment_results.extend(batch_sentiments)

            for review in batch:
                emotions = emotion_pipeline(review)[0]
                all_emotion_results.append({e['label'].lower(): e['score'] for e in emotions})

        # Sentiment
        df['predicted_sentiment'] = [label_map.get(r['label'], r['label']) for r in all_sentiment_results]
        df['sentiment_confidence'] = [r['score'] for r in all_sentiment_results]

        # Emotion
        df['emotion'] = all_emotion_results

        # Mismatch check
        mismatches = df[df['predicted_sentiment'] != df['rating_sentiment']]

        st.subheader("Batch Analysis Results")
        if len(mismatches) > 0:
            st.warning(f"⚠️ Found {len(mismatches)} mismatched reviews")
            st.dataframe(mismatches[['review', 'rating', 'rating_sentiment', 'predicted_sentiment', 'sentiment_confidence', 'emotion']])
        else:
            st.success("✅ All reviews match rating sentiment.")

