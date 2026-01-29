# =========================
# ENV FIXES
# =========================
import os
os.environ["TORCH_DISABLE_SDPA"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from stqdm import stqdm

# =========================
# CACHE MODELS
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
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

sentiment_pipeline = load_sentiment_model()
emotion_tokenizer, emotion_model = load_emotion_model()

# =========================
# HELPERS
# =========================
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

def rating_to_sentiment(r):
    if pd.isna(r):
        return None
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    else:
        return "negative"

def get_dominant_emotion(text: str) -> str:
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
    top_idx = torch.argmax(probs).item()
    return labels[top_idx].lower()

# =========================
# STREAMLIT UI
# =========================
st.title("🍔 Restaurant Review Sentiment & Emotion Dashboard")

# =====================================================
# SINGLE REVIEW ANALYSIS
# =====================================================
st.header("📝 Single Review Analysis")

review = st.text_area("Enter your review")
rating = st.radio(
    "Rate the restaurant",
    [1, 2, 3, 4, 5],
    format_func=lambda x: "⭐" * x,
    horizontal=True
)

if st.button("Analyze Single Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # Sentiment
    sent = sentiment_pipeline(review)[0]
    sentiment_label = label_map.get(sent["label"], sent["label"])
    sentiment_score = sent["score"]

    # Emotions
    emotion_dict = get_emotions = {k.lower(): v for k, v in get_emotions_single(review).items()} if 'get_emotions_single' in locals() else {get_dominant_emotion(review): 1.0}

    st.subheader("Sentiment Result")
    st.write(f"**Sentiment:** {sentiment_label}")
    st.write(f"**Confidence:** {sentiment_score:.2f}")
    st.write(f"**Rating Sentiment:** {rating_to_sentiment(rating)}")

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

# =====================================================
# CSV BATCH ANALYSIS
# =====================================================
st.header("📂 Batch Review Analysis (CSV Upload)")

file = st.file_uploader(
    "Upload CSV with columns: review, rating",
    type=["csv"]
)

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
    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    if st.button("Analyze CSV Reviews"):
        reviews = df["review"].astype(str).tolist()

        # ---------------------------
        # SENTIMENT ANALYSIS
        # ---------------------------
        labels, scores = [], []
        BATCH = 16

        with st.spinner("Running sentiment analysis..."):
            for i in stqdm(range(0, len(reviews), BATCH)):
                batch = reviews[i:i+BATCH]
                try:
                    results = sentiment_pipeline(batch)
                except:
                    results = [{"label": "NEU", "score": 0.0}] * len(batch)

                for r in results:
                    labels.append(label_map.get(r["label"], r["label"]))
                    scores.append(r["score"])

        df["predicted_sentiment"] = labels
        df["sentiment_confidence"] = scores

        # =========================
        # SUMMARY BOX
        # =========================
        mismatches = df[df["predicted_sentiment"] != df["rating_sentiment"]]

        st.subheader("📌 Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", len(df))
        c2.metric("Mismatches", len(mismatches))
        c3.metric("Mismatch Rate", f"{(len(mismatches)/len(df))*100:.1f}%")

        # =========================
        # OVERALL CHARTS
        # =========================
        st.subheader("📊 Overall Analysis")

        # Sentiment Pie
        sentiment_counts = df["predicted_sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig_overall_pie = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            title="Overall Sentiment Distribution"
        )
        st.plotly_chart(fig_overall_pie, use_container_width=True)

        # Emotion Bar
        with st.spinner("Detecting emotions (overall)..."):
            df["dominant_emotion"] = [
                get_dominant_emotion(text)
                for text in stqdm(df["review"].astype(str))
            ]

        emotion_counts = df["dominant_emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        emotion_counts["Emotion"] = emotion_counts["Emotion"].apply(
            lambda x: f"{emoji_map.get(x,'')} {x.capitalize()}"
        )

        fig_overall_bar = px.bar(
            emotion_counts,
            x="Count",
            y="Emotion",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Viridis",
            title="Overall Dominant Emotions"
        )
        fig_overall_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_overall_bar, use_container_width=True)

        # =========================
        # MISMATCH CHARTS
        # =========================
        if not mismatches.empty:
            st.subheader("⚠️ Mismatches Analysis")
            mismatches["dominant_emotion"] = df.loc[mismatches.index, "dominant_emotion"]

            # Mismatch Sentiment Pie
            mismatch_sentiment_counts = mismatches["predicted_sentiment"].value_counts().reset_index()
            mismatch_sentiment_counts.columns = ["Sentiment", "Count"]
            fig_mismatch_pie = px.pie(
                mismatch_sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Mismatch Sentiment Distribution"
            )
            st.plotly_chart(fig_mismatch_pie, use_container_width=True)

            # Mismatch Emotion Bar
            mismatch_emotion_counts = mismatches["dominant_emotion"].value_counts().reset_index()
            mismatch_emotion_counts.columns = ["Emotion", "Count"]
            mismatch_emotion_counts["Emotion"] = mismatch_emotion_counts["Emotion"].apply(
                lambda x: f"{emoji_map.get(x,'')} {x.capitalize()}"
            )

            fig_mismatch_bar = px.bar(
                mismatch_emotion_counts,
                x="Count",
                y="Emotion",
                orientation="h",
                text="Count",
                color="Count",
                color_continuous_scale="Viridis",
                title="Dominant Emotions (Mismatches)"
            )
            fig_mismatch_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_mismatch_bar, use_container_width=True)

            # MISMATCH TABLES
            st.subheader("Mismatches by Sentiment")
            for sentiment in ["positive", "neutral", "negative"]:
                group = mismatches[mismatches["predicted_sentiment"] == sentiment]
                with st.expander(f"{sentiment.capitalize()} Mismatches ({len(group)})"):
                    st.dataframe(group)

        st.success("✅ Analysis completed successfully!")
