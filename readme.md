# Programming Problem Difficulty Predictor

## Overview
Online coding platforms classify programming problems by difficulty level and assign a numerical difficulty score.  
This project builds an automated system that predicts:

- **Problem Difficulty Class** (Easy / Medium / Hard)
- **Problem Difficulty Score** (Numerical on scale of 1-10)

using **only textual problem descriptions**.

---

## Dataset-Raw
Each problem contains:
- Title
- Problem Description
- Input Description
- Output Description
- Problem Class (Easy / Medium / Hard)
- Problem Score (Numeric)

The dataset is pre-labeled and provided.

---

## Approach

### Data Preprocessing
- Combined all textual fields into a single text input
- Lowercasing and removal of special characters
- Handled missing values

### Feature Extraction
- TF-IDF vectorization (unigrams + bigrams)
- Additional numerical features:
  - Text length
  - Word count
  - Math Symbol count
  - Key words

### Models

#### Classification
- Logistic Regression
- Metric: Accuracy

#### Regression
- Random Forest Regressor
- Metrics: MAE, RMSE

---

## Web Interface
A simple Streamlit app allows users to:
- Paste problem description, input, and output
- Click predict
- View predicted difficulty class and score

---

## How to Run

```bash

pip install -r requirements.txt
streamlit run app.py
