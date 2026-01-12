import streamlit as st
import joblib
import re
import time

# 1. Page Config (Must be the first command)
st.set_page_config(
    page_title="Cinema Pulse AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Load Models with Caching (Makes the app faster)
@st.cache_resource
def load_models():
    try:
        vec = joblib.load('vectorizer.pkl')
        mod = joblib.load('model.pkl')
        return vec, mod
    except FileNotFoundError:
        return None, None

vectorizer, model = load_models()

# 3. Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

# 4. Sidebar UI
with st.sidebar:
    st.title("ℹ️ About")
    st.info(
        """
        This AI model was trained on **50,000 movie reviews** using 
        Logistic Regression and TF-IDF Vectorization.
        """
    )
    st.write("---")
    st.write("**Created by:** Rahul")
    st.write("**Tech Stack:** Python, Scikit-Learn, Streamlit")

# 5. Main UI
if vectorizer is None or model is None:
    st.error("⚠️ Error: Model files not found. Please upload 'model.pkl' and 'vectorizer.pkl'.")
else:
    # Title with custom color
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Cinema Pulse AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Analyze the sentiment of any movie review instantly.</p>", unsafe_allow_html=True)
    
    st.write("---")

    # Text Input
    user_input = st.text_area("✍️ Type or paste a review here:", height=150, placeholder="Example: The movie was absolutely fantastic! The acting was great...")

    # Analyze Button (Centered using columns)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)

    # Prediction Logic
    if analyze_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1) # Added a small delay for dramatic effect
                
                # Process
                cleaned_review = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned_review])
                prediction = model.predict(vec_input)[0]
                probability = model.predict_proba(vec_input)[0].max()

                # Results Display
                st.write("---")
                st.subheader("🎯 Analysis Result:")
                
                if prediction == 'positive':
                    st.balloons()  # YOUR FAVORITE FEATURE 🎈
                    st.success(f"**Positive Review**")
                    st.metric(label="Confidence Score", value=f"{probability:.1%}", delta="High Confidence")
                else:
                    st.snow()  # NEW FEATURE ❄️
                    st.error(f"**Negative Review**")
                    st.metric(label="Confidence Score", value=f"{probability:.1%}", delta="-Negative Sentiment")
                
                # Visual Progress Bar
                st.write("Confidence Level:")
                st.progress(probability)

    # How it works section
    st.write("---")
    with st.expander("🤔 How does this work?"):
        st.write("""
        1. **Text Cleaning:** The AI removes punctuation and HTML tags.
        2. **Vectorization:** It converts words into numbers using TF-IDF.
        3. **Prediction:** The Logistic Regression model calculates the probability.
        """)