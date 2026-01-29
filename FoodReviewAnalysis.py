# =========================
# ENV FIXES (STREAMLIT CLOUD)
# =========================
import os
os.environ["TORCH_DISABLE_SDPA"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import streamlit as st
import pandas as pd
from transformers import pipeline
from stqdm import stqdm
import plotly.express as px

# =========================
# CACHE MODELS (CRITICAL)
# =========================
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,
        truncation=True,
        max_length=512
    )

@st.cache_resource(show_spinner="Loading emotion model...")
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
        device=-1,
        truncation=True,
        max_length=512
    )

sentiment_pipeline = load_sentiment_model()
emotion_pipeline = load_emotion_model()

label_map = {"POS": "positive", "NEU": "neutral", "NEG": "negative"}

emoji_map = {
    "joy": "😊",
    "anger": "😡",
    "sadness": "😢",
    "fear": "😱",
    "surprise": "😲",
    "love": "❤️",
    "neutral": "😐",
    "disgust": "😒"
}

# =========================
# UI
# =========================
st.title("🍔 Restaurant Review Sentiment Dashboard")

# =========================
# SINGLE REVIEW
# =========================
st.subheader("Single Review Analysis")

review = st.text_area("Enter review text")
rating = st.radio("Rate the restaurant", [1,2,3,4,5], horizontal=True)

if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # Sentiment
    sent = sentiment_pipeline(review)[0]
    sentiment_label = label_map.get(sent["label"], sent["label"])
    sentiment_score = sent["score"]

    # Emotion (FIXED)
    emotion_raw = emotion_pipeline(review)

    # 🔑 normalize output
    if isinstance(emotion_raw, list) and isinstance(emotion_raw[0], list):
        emotion_raw = emotion_raw[0]

    emotion_dict = {
        e["label"].lower(): float(e["score"])
        for e in emotion_raw
    }

    # Display
    st.subheader("Sentiment Result")
    st.write(f"**Sentiment:** {sentiment_label}")
    st.write(f"**Confidence:** {sentiment_score:.2f}")

    st.subheader("Emotion Breakdown")

    df_emotion = pd.DataFrame({
        "Emotion": [f"{emoji_map.get(k,'')} {k.capitalize()}" for k in emotion_dict],
        "Score": [v * 100 for v in emotion_dict.values()]
    }).sort_values("Score")

    fig = px.bar(
        df_emotion,
        x="Score",
        y="Emotion",
        orientation="h",
        text="Score",
        color="Score",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)

# =========================
# CSV ANALYSIS
# =========================
st.subheader("Batch Review Analysis (CSV)")

file = st.file_uploader("Upload CSV with columns: review, rating", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_csv(file, encoding="latin1")

    if not {"review", "rating"}.issubset(df.columns):
        st.error("CSV must contain 'review' and 'rating' columns.")
        st.stop()

    MAX_ROWS = 300
    if len(df) > MAX_ROWS:
        st.warning(f"Only first {MAX_ROWS} rows will be analyzed.")
        df = df.head(MAX_ROWS)

    df["rating"] = df["rating"].astype(str).str.extract(r"(\d)").astype(float)

    def rating_to_sentiment(r):
        if r >= 4: return "positive"
        if r == 3: return "neutral"
        return "negative"

    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    if st.button("Analyze CSV Reviews"):
        reviews = df["review"].astype(str).tolist()

        labels, scores = [], []
        BATCH = 16

        with st.spinner("Analyzing reviews…"):
            for i in stqdm(range(0, len(reviews), BATCH)):
                batch = reviews[i:i+BATCH]
                try:
                    results = sentiment_pipeline(batch)
                except:
                    results = [{"label":"NEU","score":0.0}] * len(batch)

                for r in results:
                    labels.append(label_map.get(r["label"], r["label"]))
                    scores.append(r["score"])

        df["predicted_sentiment"] = labels
        df["sentiment_confidence"] = scores

        mismatches = df[df["predicted_sentiment"] != df["rating_sentiment"]]

        st.subheader("Results")
        st.metric("Total Reviews", len(df))
        st.metric("Mismatches", len(mismatches))

        if len(mismatches) > 0:
            st.warning("⚠️ Mismatched Reviews")
            st.dataframe(mismatches)
        else:
            st.success("✅ No mismatches found")

        st.subheader("Full Output")
        st.dataframe(df)
