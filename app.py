import streamlit as st
import pickle
import pandas as pd
import streamlit.components.v1 as components

# Initialize session state for prediction
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Load the trained model and label encoder
try:
    with open('personality_prediction.pkl', 'rb') as file:
        model_payload = pickle.load(file)
    model = model_payload['model']
    le = model_payload['label_encoder']
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# Personality descriptions
personality_descriptions = {
    "extraverted": "You are likely outgoing, sociable, and energetic. You enjoy being around others and are often the life of the party. You thrive in social settings and feel energized by interacting with people.",
    "serious": "You tend to be thoughtful, disciplined, and goal-oriented. You approach tasks with a structured and organized mindset, and you value responsibility and reliability in yourself and others.",
    "dependable": "You are reliable, trustworthy, and responsible. People can count on you to follow through on your commitments. You are practical and well-organized, making you a cornerstone in any team or family.",
    "lively": "You are enthusiastic, cheerful, and full of energy. You bring a positive and vibrant attitude to everything you do. Your spontaneity and optimism are contagious, and you enjoy new and exciting experiences.",
    "responsible": "You are conscientious, diligent, and accountable for your actions. You have a strong sense of duty and take your obligations seriously. You are a planner and prefer to be prepared for all outcomes."
}

personality_icons = {
    "extraverted": "🎉",
    "serious": "🤔",
    "dependable": "🤝",
    "lively": "🥳",
    "responsible": "📋"
}


# Streamlit app layout
st.set_page_config(page_title="Personality Insights", layout="wide", page_icon="🧠")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: #f0f8ff;
    }
    .stApp {
        background: #f0f8ff;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #ff4b4b;
        border-radius: 25px;
        border: 2px solid #ff4b4b;
        padding: 12px 28px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.4s;
        width: 100%;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #ffffff;
        color: #ff4b4b;
        border-color: #ff4b4b;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    .stSlider > div > div > div > div {
        color: #2c3e50;
        font-weight: 600;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    .result-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 30px;
        margin-top: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 7px solid #ff4b4b;
        text-align: center;
    }
    .input-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 4rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("https://www.16personalities.com/static/images/brand/logotype/main.svg", width=250)
    st.header("About the Big Five Model")
    st.write("This app uses the **Big Five personality model** (the inputs) to predict a personality **trait** (the output).")
    
    st.subheader("Input Traits (The Big Five)")
    with st.expander("🌍 Openness"):
        st.write("Measures imagination, creativity, and intellectual curiosity. High scorers are inventive and curious.")
    with st.expander("🎯 Conscientiousness"):
        st.write("Measures organization, thoroughness, and responsibility. High scorers are efficient and organized.")
    with st.expander("🎉 Extraversion"):
        st.write("Measures sociability, assertiveness, and emotional expression. High scorers are outgoing and energetic.")
    with st.expander("🤝 Agreeableness"):
        st.write("Measures compassion, cooperation, and trustworthiness. High scorers are friendly and compassionate.")
    with st.expander("🧠 Neuroticism"):
        st.write("Measures emotional stability and the tendency to experience negative emotions. High scorers are sensitive and nervous.")
    
    # Dynamically show the predicted trait description
    if st.session_state.prediction:
        predicted_trait = st.session_state.prediction
        icon = personality_icons.get(predicted_trait, "💡")
        description = personality_descriptions.get(predicted_trait, "No description available.")
        
        st.subheader("Your Predicted Trait")
        with st.expander(f"{icon} {predicted_trait.capitalize()}", expanded=True):
            st.write(description)

    st.info("Your data is not saved. This app is for educational purposes only.")

# --- Main App ---
st.title("Discover Your Personality Insight")
st.write("How do you see yourself? Answer these questions on a scale of 1 to 10 to reveal a key aspect of your personality.")

if not model_loaded:
    st.error("Model file 'personality_prediction.pkl' not found. Please run `model.py` to train and save the model first.")
else:
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            openness = st.slider("Openness: I have a vivid imagination and love new experiences.", 1, 10, 5)
            conscientiousness = st.slider("Conscientiousness: I am organized, disciplined, and pay attention to detail.", 1, 10, 5)
            extraversion = st.slider("Extraversion: I am talkative, outgoing, and the life of the party.", 1, 10, 5)
        
        with col2:
            agreeableness = st.slider("Agreeableness: I am sympathetic, warm, and considerate of others.", 1, 10, 5)
            neuroticism = st.slider("Neuroticism: I can be moody, tense, and get stressed easily.", 1, 10, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("✨ Reveal My Personality Trait"):
        user_input = pd.DataFrame({
            'openness': [openness],
            'neuroticism': [neuroticism],
            'conscientiousness': [conscientiousness],
            'agreeableness': [agreeableness],
            'extraversion': [extraversion]
        })
        
        prediction_encoded = model.predict(user_input)
        prediction_label = le.inverse_transform(prediction_encoded)[0]
        # Store the prediction in the session state
        st.session_state.prediction = prediction_label
    
    # Display the result card if a prediction has been made
    if st.session_state.prediction:
        prediction_label = st.session_state.prediction
        icon = personality_icons.get(prediction_label, "💡")
        
        st.markdown(f"""
        <div class="result-card">
            <h1 style="font-size: 5rem; margin: 0;">{icon}</h1>
            <h3>Your predicted personality trait is:</h3>
            <h2 style="color: #ff4b4b; font-size: 2.5rem;">{prediction_label.capitalize()}</h2>
            <p style="font-size: 1.1em; max-width: 600px; margin: auto;">{personality_descriptions.get(prediction_label, "No description available.")}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr><center>© Made with Streamlit for Personality Prediction</center>", unsafe_allow_html=True)

