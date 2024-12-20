import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Neural Network Library",
    page_icon="🧠",
    layout="wide"
)

# CSS for custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e2e;
        color: #ffffff;
        font-family: Arial, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #ff79c6;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        color: #bd93f9;
    }
    .content-text {
        text-align: center;
        font-size: 1.1rem;
        margin-top: 10px;
        color: #ffffff;
    }
    .task-button {
        background-color: #44475a;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        color: #ffffff;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease-in-out;
        width: 100%;
        height: 150px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .task-button:hover {
        background-color: #6272a4;
        transform: translateY(-5px);
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 1rem;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Subtitle
st.markdown("<h1 class='main-title'>Neural Network Library 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Build, train, and visualize neural networks with ease!</p>", unsafe_allow_html=True)
st.markdown("<p class='content-text'>Welcome to the Neural Network Library! This tool allows you to create and train neural networks<br>for various tasks, including regression, classification, and image data processing.</p>", unsafe_allow_html=True)

# Task Buttons
st.markdown("<h2 style='text-align: center; color: #bd93f9;'>Select Your Task</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

# Button navigation with page redirection using st.switch_page
with col1:
    if st.button("🧮 Regression\nPerform regression tasks on your data", key="regression", help="Go to regression tasks"):
        st.switch_page("pages/regressionpage.py")

with col2:
    if st.button("📊 Classification\nClassify data into different categories", key="classification", help="Go to classification tasks"):
        st.switch_page("pages/classificationpage.py")

with col3:
    if st.button("💬 NLP Tasks\nWork on text-based tasks and models", key="nlp", help="Go to NLP tasks"):
        st.switch_page("pages/nlpPage.py")

# Footer divider and credits
st.markdown("---")
st.markdown(
    "<p class='footer'>Developed with ❤️ for AI enthusiasts.</p>",
    unsafe_allow_html=True
)
