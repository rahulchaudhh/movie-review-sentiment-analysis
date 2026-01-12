import streamlit as st
import joblib
import re

# Load the saved model and vectorizer
try:
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found. Please verify they are in the same folder as this script.")
    st.stop()

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

# UI Layout
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to see if it's **Positive** or **Negative**.")

# Input Area
user_input = st.text_area("Type your review here:", height=150)

# Prediction Logic
if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_review = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_review])
        
        prediction = model.predict(vec_input)[0]
        probability = model.predict_proba(vec_input)[0].max()
        
        if prediction == 'positive':
            st.success(f"**Positive Review** (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.error(f"**Negative Review** (Confidence: {probability:.1%})")
    else:
        st.warning("Please enter some text first.")
