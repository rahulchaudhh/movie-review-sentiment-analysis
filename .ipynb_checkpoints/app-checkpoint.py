import streamlit as st
import joblib
import re

# 1. Load the saved model and vectorizer
# (Make sure these files are in the same folder as this script)
# We use a try-except block just in case files are missing locally
try:
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the repository.")
    st.stop()

# 2. Define the cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
    return text

# 3. Build the UI
st.set_page_config(page_title="Movie Sentiment AI", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to see if it's **Positive** or **Negative**.")

# Text Input
user_input = st.text_area("Type your review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess input
        cleaned_review = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_review])
        
        # Predict
        prediction = model.predict(vec_input)[0]
        probability = model.predict_proba(vec_input)[0].max()
        
        # Display Result
        if prediction == 'positive':
            st.success(f"**Positive Review** (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.error(f"**Negative Review** (Confidence: {probability:.1%})")
    else:
        st.warning("Please enter some text first.")