# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stqdm import stqdm

# ---------------------------
# Load Sentiment Model (SAFE) - WITH CACHING
# ---------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            SENTIMENT_MODEL,
            clean_up_tokenization_spaces=True
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            SENTIMENT_MODEL,
            torch_dtype=torch.float32
        )
        
        # Move model to CPU explicitly
        model.to("cpu")
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to a simpler model if needed
        fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
        model.to("cpu")
        model.eval()
        return tokenizer, model

# Load model and tokenizer
sentiment_tokenizer, sentiment_model = load_model_and_tokenizer()

LABELS = ["negative", "neutral", "positive"]

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    if not text or str(text).strip() == "":
        return "neutral", 0.5
    
    try:
        inputs = sentiment_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move inputs to CPU (model is already on CPU)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        
        idx = torch.argmax(probs).item()
        return LABELS[idx], round(probs[0][idx].item(), 4)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "neutral", 0.5

# ---------------------------
# Rating → Sentiment
# ---------------------------
def rating_to_sentiment(x):
    if pd.isna(x):
        return None
    try:
        x = float(x)
        if x >= 4:
            return "positive"
        elif x == 3:
            return "neutral"
        elif x <= 2:
            return "negative"
        else:
            return "neutral"
    except (ValueError, TypeError):
        return "neutral"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(
    page_title="McDonald's Review Sentiment Dashboard",
    page_icon="🍔",
    layout="wide"
)

st.title("🍔 McDonald's Review Sentiment Dashboard")
st.write("Analyze sentiment and compare it with rating.")

# ---------- Single Review ----------
with st.expander("🔍 Single Review Analysis", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        user_review = st.text_area(
            "Enter a review:",
            height=150,
            placeholder="Type your McDonald's review here..."
        )
    
    with col2:
        user_rating = st.number_input(
            "Enter rating (1–5):",
            min_value=1,
            max_value=5,
            value=5,
            step=1,
            help="1 = Very poor, 5 = Excellent"
        )
    
    if st.button("Analyze Review", type="primary"):
        if user_review and user_review.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment_label, sentiment_score = predict_sentiment(user_review)
                rating_sentiment = rating_to_sentiment(user_rating)
            
            st.subheader("📊 Results")
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Sentiment", 
                    sentiment_label,
                    f"{sentiment_score:.1%} confidence"
                )
            
            with col2:
                st.metric("Rating Sentiment", rating_sentiment)
            
            with col3:
                match = sentiment_label == rating_sentiment
                status = "✅ Match" if match else "⚠️ Mismatch"
                st.metric("Status", status)
            
            # Progress bars for confidence
            st.write("**Confidence Levels:**")
            
            # Get all probabilities for visualization
            if user_review.strip():
                try:
                    inputs = sentiment_tokenizer(
                        user_review,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = sentiment_model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                    
                    # Display probability bars
                    for label, prob in zip(LABELS, probs[0]):
                        st.progress(
                            float(prob),
                            text=f"{label.capitalize()}: {prob:.1%}"
                        )
                except:
                    pass
        else:
            st.warning("Please enter a review to analyze.")

# ---------- CSV Upload ----------
with st.expander("📁 Batch Review Analysis (CSV)"):
    st.info("Upload a CSV file with 'review' and 'rating' columns.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV should contain at least 'review' and 'rating' columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1")
        
        st.success(f"✅ Loaded {len(df)} reviews")
        
        # Show preview
        with st.expander("Preview Data"):
            st.dataframe(df.head(), use_container_width=True)
        
        # Check required columns
        required_cols = ["review", "rating"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Clean rating column
        st.write("**Cleaning rating column...**")
        df["rating_clean"] = (
            df["rating"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")[0]  # Extract any number
            .astype(float)
        )
        
        df["rating_sentiment"] = df["rating_clean"].apply(rating_to_sentiment)
        
        # Remove rows with missing reviews
        df = df.dropna(subset=["review"])
        
        if st.button("Analyze All Reviews", type="primary"):
            predictions = []
            confidence_scores = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, review in enumerate(stqdm(df["review"].astype(str), desc="Analyzing reviews")):
                label, score = predict_sentiment(review)
                predictions.append(label)
                confidence_scores.append(score)
                
                # Update progress
                progress_bar.progress((i + 1) / len(df))
                status_text.text(f"Processed {i + 1}/{len(df)} reviews")
            
            df["predicted_sentiment"] = predictions
            df["confidence"] = confidence_scores
            
            # Calculate mismatches
            df["match"] = df["predicted_sentiment"] == df["rating_sentiment"]
            mismatches = df[~df["match"]]
            
            st.subheader("📈 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                match_percent = (len(df) - len(mismatches)) / len(df) * 100
                st.metric(
                    "Match Rate", 
                    f"{match_percent:.1f}%",
                    f"{len(df) - len(mismatches)}/{len(df)} reviews"
                )
            
            with col2:
                st.metric(
                    "Mismatches Found", 
                    len(mismatches)
                )
            
            with col3:
                avg_confidence = df["confidence"].mean()
                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.1%}"
                )
            
            # Show mismatches in expander
            if len(mismatches) > 0:
                with st.expander(f"🔍 View {len(mismatches)} Mismatches", expanded=False):
                    st.dataframe(
                        mismatches[[
                            "review", 
                            "rating_clean", 
                            "rating_sentiment", 
                            "predicted_sentiment",
                            "confidence"
                        ]].rename(columns={
                            "rating_clean": "rating",
                            "predicted_sentiment": "model_sentiment"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download mismatches
                    csv = mismatches.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Mismatches as CSV",
                        data=csv,
                        file_name="mismatched_reviews.csv",
                        mime="text/csv"
                    )
            else:
                st.success("🎉 Perfect match! All sentiment predictions align with ratings.")
            
            # Download full results
            st.divider()
            st.write("**Download Full Results**")
            full_csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results",
                data=full_csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

# ---------- Footer ----------
st.divider()
st.caption("""
    **Note**: This tool uses the `twitter-roberta-base-sentiment-latest` model 
    from CardiffNLP for sentiment analysis. Ratings are converted to sentiment 
    as follows: 1-2 = Negative, 3 = Neutral, 4-5 = Positive.
""")
