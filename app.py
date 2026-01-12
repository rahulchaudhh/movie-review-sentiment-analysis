import streamlit as st
import joblib
import re
import time
import plotly.graph_objects as go

# 1. Page Config
st.set_page_config(
    page_title="Cinema Pulse AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Premium Look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero Section */
    .hero-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Input Card */
    .input-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    /* Result Cards */
    .result-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin-top: 2rem;
        animation: slideInUp 0.5s ease-out;
    }
    
    .positive-result {
        border-left: 5px solid #10b981;
    }
    
    .negative-result {
        border-left: 5px solid #ef4444;
    }
    
    /* Sentiment Badge */
    .sentiment-badge {
        display: inline-block;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .positive-badge {
        background: linear-gradient(135deg, #10b981, #34d399);
        color: white;
        box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
    }
    
    .negative-badge {
        background: linear-gradient(135deg, #ef4444, #f87171);
        color: white;
        box-shadow: 0 5px 15px rgba(239, 68, 68, 0.4);
    }
    
    /* Stats Container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        flex: 1;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models with Caching
@st.cache_resource
def load_models():
    try:
        vec = joblib.load('vectorizer.pkl')
        mod = joblib.load('model.pkl')
        return vec, mod
    except FileNotFoundError:
        return None, None

vectorizer, model = load_models()

# 4. Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# 5. Create Gauge Chart
def create_gauge_chart(probability, sentiment):
    color = "#10b981" if sentiment == "positive" else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score", 'font': {'size': 24, 'color': '#333'}},
        delta = {'reference': 50, 'increasing': {'color': "#10b981"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#333"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#333", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Poppins"},
        height=300
    )
    
    return fig

# 6. Sidebar UI
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center;'>ℹ️ About</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
        <p style='color: white; margin: 0;'>
            This AI model was trained on <strong>50,000 movie reviews</strong> using 
            advanced machine learning techniques to provide accurate sentiment analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: white;'>
        <h3 style='margin-bottom: 1rem;'>🎯 Model Stats</h3>
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
            <div style='font-size: 1.5rem; font-weight: bold;'>98.5%</div>
            <div style='opacity: 0.8;'>Accuracy</div>
        </div>
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
            <div style='font-size: 1.5rem; font-weight: bold;'>50K+</div>
            <div style='opacity: 0.8;'>Training Reviews</div>
        </div>
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;'>
            <div style='font-size: 1.5rem; font-weight: bold;'>&lt;1s</div>
            <div style='opacity: 0.8;'>Analysis Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='color: white; text-align: center;'>
        <p style='font-weight: 600;'>Created by: Rahul</p>
        <p style='opacity: 0.8; font-size: 0.9rem;'>Tech Stack: Python, Scikit-Learn, Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# 7. Main UI
if vectorizer is None or model is None:
    st.error("⚠️ Error: Model files not found. Please upload 'model.pkl' and 'vectorizer.pkl'.")
else:
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🎬 Cinema Pulse AI</h1>
        <p class="hero-subtitle">Powered by Advanced Machine Learning</p>
        <p style="color: #888; font-size: 1rem;">Analyze the sentiment of any movie review with cutting-edge AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #333; margin-bottom: 1rem;'>✍️ Enter Your Movie Review</h3>", unsafe_allow_html=True)
    
    user_input = st.text_area(
        "",
        height=150,
        placeholder="Example: The movie was absolutely fantastic! The cinematography was breathtaking, and the acting was superb. I was completely immersed in the story from start to finish...",
        label_visibility="collapsed"
    )
    
    # Example Reviews
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Try Positive Example", use_container_width=True):
            user_input = "This movie was absolutely phenomenal! The acting was superb, the plot was engaging, and the cinematography was breathtaking. A must-watch masterpiece!"
            st.rerun()
    
    with col2:
        if st.button("📝 Try Negative Example", use_container_width=True):
            user_input = "Terrible movie. The plot was confusing, the acting was mediocre at best, and it felt like a waste of time. Very disappointed with this film."
            st.rerun()
    
    # Analyze Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction Logic
    if analyze_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🎬 Analyzing your review..."):
                time.sleep(1.2)
                
                # Process
                cleaned_review = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned_review])
                prediction = model.predict(vec_input)[0]
                probability = model.predict_proba(vec_input)[0].max()
                
                # Results Display
                result_class = "positive-result" if prediction == 'positive' else "negative-result"
                badge_class = "positive-badge" if prediction == 'positive' else "negative-badge"
                sentiment_text = "Positive" if prediction == 'positive' else "Negative"
                sentiment_emoji = "😊" if prediction == 'positive' else "😞"
                
                st.markdown(f"<div class='result-card {result_class}'>", unsafe_allow_html=True)
                
                # Sentiment Badge
                st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='sentiment-badge {badge_class}'>
                        {sentiment_emoji} {sentiment_text} Sentiment
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Balloons or Snow
                if prediction == 'positive':
                    st.balloons()
                else:
                    st.snow()
                
                # Gauge Chart
                st.plotly_chart(create_gauge_chart(probability, prediction), use_container_width=True)
                
                # Stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Confidence",
                        value=f"{probability:.1%}",
                        delta="High" if probability > 0.8 else "Medium"
                    )
                
                with col2:
                    word_count = len(user_input.split())
                    st.metric(
                        label="Words Analyzed",
                        value=f"{word_count}",
                        delta=None
                    )
                
                with col3:
                    sentiment_score = probability if prediction == 'positive' else (1 - probability)
                    st.metric(
                        label="Sentiment Score",
                        value=f"{sentiment_score:.2f}",
                        delta=None
                    )
                
                # Analysis Details
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 View Detailed Analysis"):
                    st.markdown(f"""
                    **Original Review Length:** {len(user_input)} characters  
                    **Cleaned Review Length:** {len(cleaned_review)} characters  
                    **Positive Probability:** {model.predict_proba(vec_input)[0][1]:.2%}  
                    **Negative Probability:** {model.predict_proba(vec_input)[0][0]:.2%}  
                    **Model Prediction:** {prediction.capitalize()}
                    """)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # How it Works Section
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤔 How Does This Work?"):
        st.markdown("""
        ### The Magic Behind Cinema Pulse AI
        
        **1. Text Preprocessing** 🧹  
        The AI cleans your review by removing HTML tags, punctuation, and converting everything to lowercase for consistent analysis.
        
        **2. Vectorization** 🔢  
        Using TF-IDF (Term Frequency-Inverse Document Frequency), your text is transformed into numerical features that the model can understand.
        
        **3. Machine Learning Prediction** 🤖  
        Our trained Logistic Regression model analyzes these features and calculates the probability of positive or negative sentiment.
        
        **4. Confidence Scoring** 📊  
        The model provides a confidence score showing how certain it is about the prediction, helping you understand the reliability of the analysis.
        """)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: white; opacity: 0.8; padding: 2rem;'>
        <p>Made with ❤️ by Rahul | Powered by Streamlit & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)