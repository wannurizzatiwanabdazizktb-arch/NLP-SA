# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stqdm import stqdm

# ---------------------------
# Torch safety (CPU only)
# ---------------------------
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------
# Load Sentiment Model (SAFE)
# ---------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL,
    torch_dtype=torch.float32
)
sentiment_model.eval()

LABELS = ["negative", "neutral", "positive"]

def predict_sentiment(text):
    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    idx = torch.argmax(probs).item()
    return LABELS[idx], probs[0][idx].item()

# ---------------------------
# Rating → Sentiment
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🍔 McDonald's Review Sentiment Dashboard")
st.write("Analyze sentiment and compare it with rating.")

# ---------- Single Review ----------
st.subheader("Single Review Analysis")

user_review = st.text_area("Enter a review:")
user_rating = st.number_input(
    "Enter rating (1–5):",
    min_value=1,
    max_value=5,
    value=5,
    step=1
)

if st.button("Analyze Review"):
    if user_review.strip():
        sentiment_label, sentiment_score = predict_sentiment(user_review)
        rating_sentiment = rating_to_sentiment(user_rating)

        st.subheader("Results")
        st.write(f"**Predicted Sentiment:** {sentiment_label}")
        st.write(f"**Confidence:** {sentiment_score:.2f}")
        st.write(f"**Rating Sentiment:** {rating_sentiment}")

        if sentiment_label != rating_sentiment:
            st.warning("⚠️ Sentiment does not match rating")
        else:
            st.success("✅ Sentiment matches rating")

# ---------- CSV Upload ----------
st.subheader("Batch Review Analysis (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV with 'review' and 'rating' columns",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success(f"Loaded {len(df)} reviews")
    st.dataframe(df.head())

    # Clean rating column
    df["rating"] = (
        df["rating"]
        .astype(str)
        .str.extract(r"(\d)")
        .astype(float)
    )

    df["rating_sentiment"] = df["rating"].apply(rating_to_sentiment)

    if st.button("Analyze CSV Reviews"):
        predictions = []

        for review in stqdm(df["review"].astype(str), desc="Analyzing"):
            label, _ = predict_sentiment(review)
            predictions.append(label)

        df["predicted_sentiment"] = predictions

        mismatches = df[df["predicted_sentiment"] != df["rating_sentiment"]]

        if len(mismatches) > 0:
            st.warning(f"⚠️ Found {len(mismatches)} mismatches")
            st.dataframe(
                mismatches[
                    ["review", "rating", "rating_sentiment", "predicted_sentiment"]
                ]
            )
        else:
            st.success("✅ No mismatches found")
