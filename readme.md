# Programming Problem Difficulty Predictor
## Overview

Competitive programming platforms (Kattis, Codeforces, etc.) classify problems by difficulty. This project automates that process using Machine Learning.

By analyzing the textual description of a problem, the system predicts:

Difficulty Class: Categorizes problems into Easy, Medium, or Hard

Difficulty Score: Assigns a precise numerical score (1.0 – 10.0)

## Approach & Feature Engineering

Simple word counts were insufficient. The final model uses a Hybrid Feature Set to capture both technical language and mathematical constraints.

### 1️⃣ NLP Features

TF-IDF Vectorization: Analyzes unigrams and bigrams

Logic-Aware Cleaning: Custom preprocessing preserves:

Programming keywords (if, while, for)

LaTeX-style mathematical symbols (\leq, \geq)
(These are often removed by standard cleaners)

### 2️⃣ Domain-Specific (Handcrafted) Features
Power: Detects constraints like 10^5 or 10^9 to estimate time complexity

Algorithm Density: Frequency of keywords like dp, graph, union-find, recursion

Math Symbol Count: Density of operators (+ - * /) and comparison symbols (< > =)

Constraint Complexity: Pattern detection such as N ≤ 10^5

## Model Selection (Honest Evaluation)
### 1️⃣ Classification (Easy / Medium / Hard)

We compared four models to determine which best handled the nuances of "Medium" difficulty problems.


| Model               | Accuracy | F1-Score (Medium) | Observation                                           |
| ------------------- | -------- | ----------------- | ----------------------------------------------------- |
| Random Forest       | 51%      | 0.65              | Best overall accuracy; very strong on Medium problems |
| Naive Bayes         | 48%      | 0.65              | High recall for Medium, but ignores Hard problems     |
| Logistic Regression | 46%      | 0.56              | Most balanced; best at catching Hard problems         |
| SVM (Linear)        | 45%      | 0.56              | High precision but computationally slower             |


Confusion Matrices

(Row = Actual, Column = Predicted)


      E    M    H
E   [46, 102, 5]
M   [25, 327, 37]
H   [22, 217, 42]



Logistic Regression

      E    M    H
E   [65, 73, 15]
M   [91, 215, 83]
H   [20, 160, 101]


Naive Bayes

      E    M    H
E   [12, 141, 0]
M   [7, 380, 2]
H   [12, 263, 6]

### 2️⃣ Regression (Numerical Score)

Evaluated using:

MAE (Mean Absolute Error) → lower is better

R² Score → higher is better

Model	MAE (↓ Better)	R² (↑ Better)
Random Forest Regressor	1.70	0.42
Gradient Boosting	1.71	0.40
Linear Regression	1.97	0.28

##  How to Run (VS Code Setup)
### 1️⃣ Environment Setup

Ensure Python 3.8+ is installed. In your terminal:

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it 
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLP data
python -m nltk.downloader stopwords wordnet omw-1.4
```

### 2️⃣ Training the Model

Process the raw data and generate the saved model files:

```bash
python src/train.py
```

This creates the /models folder containing your .joblib files.

### 3️⃣ Launching the App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Your browser will open at:
👉 http://localhost:8501

