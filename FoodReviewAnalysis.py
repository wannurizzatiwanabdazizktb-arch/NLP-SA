# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
from transformers import pipeline

# ---------------------------
# 1️⃣ Load dataset (optional)
# ---------------------------
# Assume df has 'review' and 'rating' columns
df = pd.read_csv("McDonald_s_Reviews.csv")  # adjust path

# ---------------------------
# 2️⃣ Initialize pipelines
# ---------------------------
# Sentiment (3-class)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=0  # set -1 if CPU
)

# Emotion detection (e.g., anger, joy, sadness)
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=0
)

# ---------------------------
# 3️⃣ Streamlit UI
# ---------------------------
st.title("🍔 McDonald's Review Sentiment Dashboard")
st.write("Analyze sentiment and emotion of reviews, and compare with rating.")

# 3a. Single review input
user_review = st.text_area("Enter a review:", "")

if st.button("Analyze"):
    if user_review.strip() != "":
        # Sentiment prediction
        sentiment_result = sentiment_pipeline(user_review)[0]
        label_map = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}
        sentiment_label = label_map.get(sentiment_result['label'], sentiment_result['label'])
        sentiment_score = sentiment_result['score']

        # Emotion prediction
        emotion_results = emotion_pipeline(user_review)[0]
        # Convert to dict: {'joy': 0.7, 'anger': 0.2, ...}
        emotion_dict = {e['label'].lower(): e['score'] for e in emotion_results}

        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")

        st.subheader("Emotion Analysis")
        for emotion, score in emotion_dict.items():
            st.write(f"{emotion.capitalize()}: {score:.2f}")

# 3b. Show confusion with rating
st.subheader("Mismatched Sentiment vs Rating")
# Map rating to sentiment
def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df['rating_sentiment'] = df['rating'].apply(rating_to_sentiment)

# Batch predict sentiment for dataset (optional: limit for speed)
if st.checkbox("Compute sentiment for dataset (may take time)"):
    reviews = df['review'].tolist()
    all_results = []
    for review in stqdm(reviews, desc="Processing reviews"):  # optional: stqdm for Streamlit progress
        result = sentiment_pipeline(review)[0]
        all_results.append(result)

    df['predicted_sentiment'] = [label_map[r['label']] for r in all_results]

    # Find mismatches
    mismatches = df[df['predicted_sentiment'] != df['rating_sentiment']]
    st.write(f"Found {len(mismatches)} mismatched reviews")
    st.dataframe(mismatches[['review', 'rating', 'rating_sentiment', 'predicted_sentiment']])
