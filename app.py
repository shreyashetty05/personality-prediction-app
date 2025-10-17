import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64

# Initialize session state
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Load the trained model and label encoder
try:
    with open("personality_prediction.pkl", "rb") as file:
        model_payload = pickle.load(file)
    model = model_payload["model"]
    le = model_payload["label_encoder"]
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Personality descriptions
personality_descriptions = {
    "extraverted": "You are likely outgoing, sociable, and energetic. You enjoy being around others and are often the life of the party. You thrive in social settings and feel energized by interacting with people.",
    "serious": "You tend to be thoughtful, disciplined, and goal-oriented. You approach tasks with a structured and organized mindset, and you value responsibility and reliability in yourself and others.",
    "dependable": "You are reliable, trustworthy, and responsible. People can count on you to follow through on your commitments. You are practical and well-organized, making you a cornerstone in any team or family.",
    "lively": "You are enthusiastic, cheerful, and full of energy. You bring a positive and vibrant attitude to everything you do. Your spontaneity and optimism are contagious, and you enjoy new and exciting experiences.",
    "responsible": "You are conscientious, diligent, and accountable for your actions. You have a strong sense of duty and take your obligations seriously. You are a planner and prefer to be prepared for all outcomes.",
}

personality_icons = {
    "extraverted": "🎉",
    "serious": "🤔",
    "dependable": "🤝",
    "lively": "🥳",
    "responsible": "📋",
}


# Define colors based on theme (black and white only, monochrome)
@st.cache_data
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            "bg": "#000000",
            "secondary_bg": "#111111",
            "glass_bg": "rgba(17, 17, 17, 0.9)",
            "text": "#ffffff",
            "text_secondary": "#cccccc",
            "primary": "#666666",
            "primary_hover": "#888888",
            "accent": "#999999",
            "card_bg": "rgba(17, 17, 17, 0.95)",
            "border": "#333333",
            "shadow": "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
        }
    else:
        return {
            "bg": "#ffffff",
            "secondary_bg": "#f5f5f5",
            "glass_bg": "rgba(255, 255, 255, 0.95)",
            "text": "#000000",
            "text_secondary": "#333333",
            "primary": "#666666",
            "primary_hover": "#444444",
            "accent": "#555555",
            "card_bg": "rgba(255, 255, 255, 0.98)",
            "border": "#dddddd",
            "shadow": "0 25px 50px -12px rgba(0, 0, 0, 0.1)",
        }


# Streamlit app layout
st.set_page_config(
    page_title="Personality Insights",
    layout="wide",
    page_icon="🧠",
)


# Custom CSS for monochrome black and white UI
def load_css():
    colors = get_theme_colors()
    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {{
            background: linear-gradient(135deg, {colors['bg']} 0%, {colors['secondary_bg']} 100%);
            font-family: 'Inter', sans-serif;
            backdrop-filter: blur(10px);
        }}
        .stApp {{
            background: transparent;
        }}
        [data-testid="stAppViewContainer"] > .main {{
            background: {colors['bg']};
        }}
        h1, h2, h3 {{
            color: {colors['text']};
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        .stText, .stMarkdown {{
            color: {colors['text']};
        }}
        .stSlider > div > div > div {{
            color: {colors['text']};
            font-weight: 500;
        }}
        .stSlider label {{
            color: {colors['text_secondary']};
        }}
        .stSlider .stMarkdown {{
            color: {colors['text']};
        }}
        .glass-card {{
            background: {colors['glass_bg']};
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid {colors['border']};
            border-radius: 24px;
            padding: 2rem;
            box-shadow: {colors['shadow']};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 35px 60px -12px rgba(0, 0, 0, 0.3);
        }}
        .input-section {{
            background: {colors['card_bg']};
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: {colors['shadow']};
            border: 1px solid {colors['border']};
            margin-bottom: 2rem;
        }}
        .result-section {{
            background: {colors['card_bg']};
            border: 1px solid {colors['primary']};
            color: {colors['text']};
            padding: 3.5rem;
            border-radius: 24px;
            text-align: center;
            box-shadow: {colors['shadow']};
            backdrop-filter: blur(10px);
        }}
        .result-section p {{
            color: {colors['text']};
        }}
        .stButton > button {{
            background: {colors['primary']};
            color: {colors['text']};
            border: none;
            border-radius: 50px;
            padding: 1rem 2.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            background: {colors['primary_hover']};
        }}
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        .stButton > button:hover::before {{
            left: 100%;
        }}
        .stExpander > div > label {{
            color: {colors['text']};
            font-weight: 500;
        }}
        .stExpander > div > div {{
            color: {colors['text_secondary']};
        }}
        .chart-container {{
            background: {colors['card_bg']};
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid {colors['border']};
            box-shadow: {colors['shadow']};
        }}
        .theme-toggle {{
            position: fixed;
            top: 1.5rem;
            right: 1.5rem;
            z-index: 1000;
            background: {colors['card_bg']};
            border: 1px solid {colors['border']};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: {colors['shadow']};
            transition: all 0.3s ease;
            color: {colors['text']};
            font-size: 1.2rem;
        }}
        .theme-toggle:hover {{
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }}
        .metric-container {{
            background: {colors['primary']};
            color: {colors['text']};
            padding: 1rem;
            border-radius: 16px;
            text-align: center;
        }}
        .metric-container h3 {{
            color: {colors['text']} !important;
            margin: 0;
            font-size: 2rem;
        }}
        .metric-container p {{
            color: {colors['text']} !important;
            margin: 0;
            opacity: 0.9;
        }}
        .stInfo {{
            background: {colors['card_bg']};
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
        }}
        .stSuccess {{
            background: {colors['card_bg']};
            color: {colors['text']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
        }}
        .stError {{
            background: {colors['card_bg']};
            color: {colors['text']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background: {colors['card_bg']};
            border-radius: 12px;
            padding: 4px;
            border: 1px solid {colors['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {colors['text_secondary']};
            background: transparent;
            border-radius: 8px;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {colors['text']};
            background: {colors['glass_bg']};
        }}
        .stTabs [aria-selected="true"] {{
            color: {colors['primary']};
            background: {colors['glass_bg']};
        }}
        .stMetric {{
            background: {colors['card_bg']};
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid {colors['border']};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Dark Mode Toggle
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()


# Header with toggle
header_col1, header_col2 = st.columns([3, 0.3])
with header_col1:
    st.markdown("# 🧠 Discover Your Personality Insight")
with header_col2:
    if st.button(
        "🌙" if not st.session_state.dark_mode else "☀️",
        key="theme_toggle",
        use_container_width=False,
    ):
        toggle_dark_mode()

load_css()

st.markdown(
    "**Rate yourself on a scale of 1-10** for each Big Five trait to uncover your dominant personality aspect. Modern insights await!"
)

if not model_loaded:
    st.error(
        "❌ Model file 'personality_prediction.pkl' not found. Please run `model.py` to train and save the model first."
    )
    st.stop()

# Tabs for better organization
tab1, tab2 = st.tabs(["📝 Take Assessment", "📈 Results & Insights"])

with tab1:
    # Input Section with glass effect
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🌟 Your Self-Assessment")
        col1, col2 = st.columns(2, gap="medium")

        with col1:
            openness = st.slider(
                "🌍 Openness: I have a vivid imagination and love new experiences.",
                1,
                10,
                5,
                help="Rate your level of creativity and openness to new ideas.",
            )
            conscientiousness = st.slider(
                "🎯 Conscientiousness: I am organized, disciplined, and pay attention to detail.",
                1,
                10,
                5,
                help="Rate your level of organization and responsibility.",
            )
            extraversion = st.slider(
                "🎉 Extraversion: I am talkative, outgoing, and the life of the party.",
                1,
                10,
                5,
                help="Rate your sociability and energy in social settings.",
            )

        with col2:
            agreeableness = st.slider(
                "🤝 Agreeableness: I am sympathetic, warm, and considerate of others.",
                1,
                10,
                5,
                help="Rate your compassion and cooperation with others.",
            )
            neuroticism = st.slider(
                "🧠 Neuroticism: I can be moody, tense, and get stressed easily.",
                1,
                10,
                5,
                help="Rate your emotional stability and tendency to stress.",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction Button with shine effect
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button(
            "🔮 **Reveal My Trait**", use_container_width=True, type="primary"
        ):
            with st.spinner("🔄 Analyzing your responses with AI precision..."):
                user_input = pd.DataFrame(
                    {
                        "openness": [openness],
                        "neuroticism": [neuroticism],
                        "conscientiousness": [conscientiousness],
                        "agreeableness": [agreeableness],
                        "extraversion": [extraversion],
                    }
                )

                prediction_encoded = model.predict(user_input)
                prediction_label = le.inverse_transform(prediction_encoded)[0]
                st.session_state.prediction = prediction_label
                st.session_state.prediction_history.append(
                    {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "trait": prediction_label,
                        "scores": {
                            "openness": openness,
                            "conscientiousness": conscientiousness,
                            "extraversion": extraversion,
                            "agreeableness": agreeableness,
                            "neuroticism": neuroticism,
                        },
                    }
                )
                st.success(
                    "✅ Analysis complete! Switch to the 'Results & Insights' tab to view your modern profile. 🎉"
                )
                st.rerun()

with tab2:
    if st.session_state.prediction:
        # Result Section with glassmorphism
        col_res, _ = st.columns([1, 3])
        with col_res:
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            prediction_label = st.session_state.prediction
            icon = personality_icons.get(prediction_label, "💡")

            st.markdown(
                f"""
                <h1 style="font-size: 5rem; margin-bottom: 1.5rem; animation: pulse 2s infinite;">{icon}</h1>
                <h2 style="font-size: 3rem; margin-bottom: 1.5rem;">Your Dominant Trait:<br><strong>{prediction_label.capitalize()}</strong></h2>
                <p style="font-size: 1.3rem; line-height: 1.7; max-width: 700px; margin: 0 auto; opacity: 0.95;">{personality_descriptions.get(prediction_label, "No description available.")}</p>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Big Five Chart in modern container
        st.markdown("### 📊 Your Big Five Profile")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        scores = pd.DataFrame(
            {
                "Trait": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
                "Score": [
                    openness,
                    conscientiousness,
                    extraversion,
                    agreeableness,
                    neuroticism,
                ],
            }
        )
        colors = get_theme_colors()
        fig = px.bar(
            scores,
            x="Trait",
            y="Score",
            title="Interactive Personality Scores",
            color="Score",
            color_continuous_scale="Greys",  # Monochrome grayscale
            labels={"Score": "Rating (1-10)"},
            text="Score",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        fig.update_layout(
            plot_bgcolor=colors["card_bg"],
            paper_bgcolor=colors["card_bg"],
            font_color=colors["text"],
            font_family="Inter, sans-serif",
            showlegend=False,
            xaxis_title="Traits",
            yaxis_title="Your Score",
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Metrics for each trait
        st.markdown("### 🎯 Quick Score Overview")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f'<div class="metric-container"><h3>{openness}</h3><p>🌍 Openness</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-container"><h3>{conscientiousness}</h3><p>🎯 Conscientiousness</p></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-container"><h3>{extraversion}</h3><p>🎉 Extraversion</p></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-container"><h3>{agreeableness}</h3><p>🤝 Agreeableness</p></div>',
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f'<div class="metric-container"><h3>{neuroticism}</h3><p>🧠 Neuroticism</p></div>',
                unsafe_allow_html=True,
            )

        # Export Options
        st.markdown("### 💾 Export Your Insights")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            text_result = f"""Personality Insight Report
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Dominant Trait: {prediction_label.capitalize()}
Description: {personality_descriptions.get(prediction_label)}
Big Five Scores:
- Openness: {openness}/10
- Conscientiousness: {conscientiousness}/10
- Extraversion: {extraversion}/10
- Agreeableness: {agreeableness}/10
- Neuroticism: {neuroticism}/10"""
            st.download_button(
                "📝 Download TXT Report",
                text_result,
                f"personality_report_{prediction_label}.txt",
                "text/plain",
                use_container_width=True,
            )

        with col_export2:
            # Generate actual PDF using reportlab
            def create_pdf():
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter
                c.setFont("Helvetica-Bold", 16)
                c.drawString(100, height - 100, "Personality Insight Report")
                c.setFont("Helvetica", 12)
                y = height - 150
                lines = text_result.split("\n")
                for line in lines:
                    c.drawString(100, y, line)
                    y -= 20
                c.save()
                buffer.seek(0)
                return buffer.getvalue()

            pdf_data = create_pdf()
            st.download_button(
                "📄 Download PDF Report",
                pdf_data,
                f"personality_report_{prediction_label}.pdf",
                "application/pdf",
                use_container_width=True,
            )
    else:
        st.info(
            "👆 Complete the assessment in the 'Take Assessment' tab to unlock your modern results!"
        )

# Footer
st.markdown("---")
st.markdown(
    f"<center>© 2025 Personality Insights | Crafted with modern magic using Streamlit</center>",
    unsafe_allow_html=True,
)
