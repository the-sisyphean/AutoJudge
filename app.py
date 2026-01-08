import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from src.preprocessing import clean_text, extract_handcrafted_features

# 1. Page Configuration
st.set_page_config(page_title="Auto Judge", page_icon="⚡", layout="centered")

# 2. Load Models and Vectorizer
@st.cache_resource
def load_assets():
    clf = joblib.load('models/classifier.joblib')
    reg = joblib.load('models/regressor.joblib')
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
    return clf, reg, tfidf

try:
    clf, reg, tfidf = load_assets()
except Exception as e:
    st.error("Model files not found! Please run 'python src/train.py' first.")
    st.stop()

# 3. User Interface
st.title("AutoJudge: CP Problem Difficulty Predictor")
st.markdown("Paste the problem details below to predict its difficulty level and numeric score.")

with st.form("problem_form"):
    col = st.columns(1)
    with col[0]:
        description = st.text_area("Problem Description", placeholder="Given an array of integers...")

    input_desc = st.text_area("Input Description", placeholder="The first line contains N...")
    output_desc = st.text_area("Output Description", placeholder="Print a single integer...")
    
    submit = st.form_submit_button("Analyze Problem")

# 4. Prediction Logic
if submit:
    if not description.strip():
        st.warning("Please provide a description.")
    else:
        # A. Preprocess
        full_text = f"⚡{description} {input_desc} {output_desc}"
        cleaned = clean_text(full_text)
        
        # B. Feature Extraction
        # Text features
        X_tfidf = tfidf.transform([cleaned])
        # Handcrafted features (must be a 2D array/matrix)
        numeric_features = extract_handcrafted_features(full_text).values.reshape(1, -1)
        
        # C. Combine
        X_final = hstack([X_tfidf, csr_matrix(numeric_features)])
        
        # D. Predict
        difficulty_idx = clf.predict(X_final)[0]
        score = reg.predict(X_final)[0]
        
        # E. Map and Display Results
        difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
        label = difficulty_map.get(difficulty_idx, "Unknown")
        
        st.divider()
        st.subheader("Results")
        
        c1, c2 = st.columns(2)
        c1.metric("Predicted Level", label)
        c2.metric("Difficulty Score", f"{score:.2f}/10")
        
        # UI Polish based on difficulty
        if label == "Easy":
            st.success("This problem looks beginner-friendly!")
        elif label == "Medium":
            st.info("This problem might require specific algorithms or data structures.")
        else:
            st.warning("Brace yourself! This looks like a complex challenge.")