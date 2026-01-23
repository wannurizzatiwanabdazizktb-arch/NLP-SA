# streamlit_sentiment_dashboard.py

import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from stqdm import stqdm
import gc

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="McDonald's Review Sentiment Dashboard",
    page_icon="🍔",
    layout="wide"
)

# ---------------------------
# Load Sentiment Model - SIMPLIFIED APPROACH
# ---------------------------
@st.cache_resource
def load_sentiment_pipeline():
    """Load sentiment analysis pipeline with error handling"""
    try:
        # Try a lightweight model first
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        # Clear cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Load pipeline with explicit CPU usage
        pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,  # Force CPU
            truncation=True,
            max_length=512
        )
        
        return pipe, model_name
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to even simpler approach
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return "vader", "vader"

# Load the pipeline
model_loader = load_sentiment_pipeline()

if isinstance(model_loader, tuple):
    sentiment_pipeline, model_name = model_loader
    MODEL_TYPE = "transformers"
else:
    MODEL_TYPE = "vader"
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_sentiment_transformers(text):
    """Predict sentiment using transformers pipeline"""
    if not text or str(text).strip() == "":
        return "neutral", 0.5
    
    try:
        result = sentiment_pipeline(text[:512])[0]  # Limit to 512 chars
        label = result['label'].lower()
        score = result['score']
        
        # Standardize labels
        if 'positive' in label:
            return "positive", score
        elif 'negative' in label:
            return "negative", score
        else:
            return "neutral", score
    except Exception as e:
        st.warning(f"Transformers error, using fallback: {str(e)[:100]}")
        return predict_sentiment_vader(text)

def predict_sentiment_vader(text):
    """Predict sentiment using VADER (fallback)"""
    if not text or str(text).strip() == "":
        return "neutral", 0.5
    
    try:
        scores = vader_analyzer.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return "positive", (compound + 1) / 2  # Normalize to 0-1
        elif compound <= -0.05:
            return "negative", (-compound + 1) / 2  # Normalize to 0-1
        else:
            return "neutral", 0.5
    except:
        return "neutral", 0.5

def predict_sentiment(text):
    """Main prediction function - chooses appropriate method"""
    if MODEL_TYPE == "transformers":
        return predict_sentiment_transformers(text)
    else:
        return predict_sentiment_vader(text)

# ---------------------------
# Rating → Sentiment
# ---------------------------
def rating_to_sentiment(x):
    """Convert numeric rating to sentiment category"""
    if pd.isna(x):
        return "neutral"
    
    try:
        x = float(x)
        if x >= 4:
            return "positive"
        elif x == 3:
            return "neutral"
        else:  # 1 or 2
            return "negative"
    except (ValueError, TypeError):
        return "neutral"

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🍔 McDonald's Review Sentiment Dashboard")
st.write(f"Using: {model_name if MODEL_TYPE == 'transformers' else 'VADER Sentiment'}")

# Show model info
with st.expander("ℹ️ Model Information", expanded=False):
    if MODEL_TYPE == "transformers":
        st.write("""
        **Model:** DistilBERT fine-tuned on SST-2
        **Pros:** Accurate, understands context
        **Cons:** Larger memory footprint
        """)
    else:
        st.write("""
        **Model:** VADER Sentiment Analysis
        **Pros:** Lightweight, rule-based, fast
        **Cons:** Less nuanced than deep learning models
        """)

# ---------- Single Review Analysis ----------
st.subheader("🔍 Single Review Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    user_review = st.text_area(
        "Enter a review:",
        height=150,
        placeholder="Example: 'The Big Mac was amazing but the service was slow...'",
        key="review_input"
    )

with col2:
    user_rating = st.number_input(
        "Enter rating (1–5):",
        min_value=1.0,
        max_value=5.0,
        value=5.0,
        step=1.0,
        help="1 = Very poor, 5 = Excellent"
    )
    
    analyze_btn = st.button(
        "Analyze Review",
        type="primary",
        use_container_width=True
    )

if analyze_btn and user_review:
    with st.spinner("Analyzing sentiment..."):
        sentiment_label, sentiment_score = predict_sentiment(user_review)
        rating_sentiment = rating_to_sentiment(user_rating)
    
    # Display results
    st.success("Analysis Complete!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Sentiment", 
            sentiment_label.upper(),
            f"{sentiment_score:.1%} confidence"
        )
    
    with col2:
        st.metric(
            "Rating Sentiment", 
            rating_sentiment.upper()
        )
    
    with col3:
        match = sentiment_label == rating_sentiment
        if match:
            st.success("✅ Match")
        else:
            st.warning("⚠️ Mismatch")
    
    # Show detailed breakdown
    if MODEL_TYPE == "vader" and user_review:
        with st.expander("View VADER Analysis Details"):
            scores = vader_analyzer.polarity_scores(str(user_review))
            st.json(scores)

# ---------- Batch CSV Analysis ----------
st.subheader("📁 Batch Review Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV should contain 'review' and 'rating' columns"
)

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        except Exception as e:
            st.error(f"Cannot read file: {e}")
            df = None
    
    if df is not None and not df.empty:
        st.success(f"✅ Loaded {len(df)} reviews")
        
        # Show preview
        with st.expander("Preview Data", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
        
        # Check for required columns
        if 'review' not in df.columns or 'rating' not in df.columns:
            st.error("CSV must contain 'review' and 'rating' columns")
        else:
            # Clean data
            df_clean = df.copy()
            df_clean['review'] = df_clean['review'].astype(str).fillna('')
            
            # Clean rating
            df_clean['rating_clean'] = (
                df_clean['rating']
                .astype(str)
                .str.extract(r'(\d+\.?\d*)', expand=False)
                .astype(float)
            )
            
            df_clean['rating_sentiment'] = df_clean['rating_clean'].apply(rating_to_sentiment)
            
            # Filter out empty reviews
            df_clean = df_clean[df_clean['review'].str.strip() != '']
            
            if len(df_clean) == 0:
                st.warning("No valid reviews found in the file.")
            else:
                if st.button("Analyze All Reviews", type="primary", use_container_width=True):
                    # Analyze reviews
                    predictions = []
                    confidences = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, review in enumerate(stqdm(df_clean['review'], desc="Analyzing")):
                        label, confidence = predict_sentiment(review)
                        predictions.append(label)
                        confidences.append(confidence)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(df_clean))
                        status_text.text(f"Processed {i + 1}/{len(df_clean)} reviews")
                    
                    # Add results to dataframe
                    df_clean['predicted_sentiment'] = predictions
                    df_clean['confidence'] = confidences
                    df_clean['match'] = df_clean['predicted_sentiment'] == df_clean['rating_sentiment']
                    
                    # Calculate metrics
                    match_count = df_clean['match'].sum()
                    mismatch_count = len(df_clean) - match_count
                    match_percentage = (match_count / len(df_clean)) * 100
                    avg_confidence = df_clean['confidence'].mean()
                    
                    # Display metrics
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Match Rate",
                            f"{match_percentage:.1f}%",
                            f"{match_count}/{len(df_clean)} reviews"
                        )
                    
                    with col2:
                        st.metric("Mismatches", mismatch_count)
                    
                    with col3:
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    
                    # Show mismatches
                    if mismatch_count > 0:
                        mismatches = df_clean[~df_clean['match']]
                        
                        with st.expander(f"🔍 View {mismatch_count} Mismatches", expanded=False):
                            st.dataframe(
                                mismatches[[
                                    'review', 
                                    'rating_clean', 
                                    'rating_sentiment',
                                    'predicted_sentiment',
                                    'confidence'
                                ]].head(50),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download mismatches
                            csv_mismatch = mismatches.to_csv(index=False)
                            st.download_button(
                                label=f"Download {mismatch_count} Mismatches",
                                data=csv_mismatch,
                                file_name="mismatched_reviews.csv",
                                mime="text/csv"
                            )
                    else:
                        st.success("🎉 Perfect! All predictions match the ratings.")
                    
                    # Download all results
                    st.divider()
                    st.write("**Download Full Results**")
                    csv_all = df_clean.to_csv(index=False)
                    st.download_button(
                        label="Download All Results",
                        data=csv_all,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )

# ---------- Footer ----------
st.divider()
st.caption("""
    **Note**: 
    - Ratings are converted to sentiment: 1-2 = Negative, 3 = Neutral, 4-5 = Positive
    - Empty reviews are skipped in batch analysis
    - For best results, ensure reviews are in English
""")

# Add requirements info
with st.expander("📋 Requirements Info", expanded=False):
    st.code("""
    # Required packages:
    streamlit
    pandas
    torch
    transformers
    vaderSentiment
    stqdm
    
    # For Streamlit Cloud deployment, add to requirements.txt:
    streamlit==1.28.0
    pandas==2.1.0
    torch==2.1.0
    transformers==4.35.0
    vaderSentiment==3.3.2
    stqdm==1.3.4
    """)
