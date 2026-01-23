# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
from transformers import pipeline
from stqdm import stqdm  # optional for progress bar in Streamlit
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ---------------------------
# 1️⃣ Pipelines
# ---------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=-1,                 # CPU ONLY
    torch_dtype=torch.float32  # VERY IMPORTANT
)

emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1,
    torch_dtype=torch.float32
)

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.title("🍔 McDonald's Review Sentiment Dashboard")
st.write("Analyze sentiment and emotion of reviews, and compare with rating.")

# --- User input ---
st.subheader("Single Review Analysis")
user_review = st.text_area("Enter a review:")
user_rating = st.number_input("Enter rating (1-5):", min_value=1, max_value=5, value=5, step=1)

if st.button("Analyze Review"):
    if user_review.strip() != "":
        # Sentiment prediction
        sentiment_result = sentiment_pipeline(user_review)[0]
        sentiment_label = label_map.get(sentiment_result['label'], sentiment_result['label'])
        sentiment_score = sentiment_result['score']

        # Emotion prediction
        emotion_results = emotion_pipeline(user_review)[0]
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_results}

        # Map rating to sentiment
        def rating_to_sentiment(rating):
            if rating >= 4:
                return "positive"
            elif rating == 3:
                return "neutral"
            else:
                return "negative"

        rating_sentiment = rating_to_sentiment(user_rating)

        # Display sentiment & emotion
        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating Sentiment:** {rating_sentiment}")

        st.subheader("Emotion Analysis")
        for emotion, score in emotion_dict.items():
            st.write(f"{emotion.capitalize()}: {score:.2f}")

        # Compare sentiment and rating
        st.subheader("Sentiment vs Rating Check")
        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Mismatch detected!")
            st.dataframe(pd.DataFrame([{
                "Review": user_review,
                "Rating": user_rating,
                "Rating Sentiment": rating_sentiment,
                "Predicted Sentiment": sentiment_label,
                "Confidence": sentiment_score
            }]))
        else:
            st.success("✅ No mismatch detected.")

# --- CSV upload ---
st.subheader("Batch Review Analysis (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV file with 'review' and 'rating' columns", type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"CSV loaded successfully with {len(df)} rows.")
    st.dataframe(df.head())

    # ---------------------------
    # ✅ FIX 1: Clean rating column
    # ---------------------------
    df['rating'] = (
        df['rating']
        .astype(str)
        .str.extract(r'(\d)')   # extract 1–5 from "5 stars"
        .astype(float)
    )

    # ---------------------------
    # ✅ FIX 2: Safe rating → sentiment mapping
    # ---------------------------
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

    # ---------------------------
    # Predict sentiment
    # ---------------------------
    if st.button("Analyze CSV Reviews"):
        reviews = df['review'].astype(str).tolist()
        all_results = []

        for review in stqdm(reviews, desc="Processing reviews"):
            result = sentiment_pipeline(review)[0]
            all_results.append(result)

        df['predicted_sentiment'] = [
            label_map.get(r['label'], r['label']) for r in all_results
        ]

        # ---------------------------
        # Find mismatches
        # ---------------------------
        mismatches = df[df['predicted_sentiment'] != df['rating_sentiment']]

        if len(mismatches) > 0:
            st.warning(f"⚠️ Found {len(mismatches)} mismatched reviews")
            st.dataframe(
                mismatches[['review', 'rating', 'rating_sentiment', 'predicted_sentiment']]
            )
        else:
            st.success("✅ No mismatches found in the dataset!")

