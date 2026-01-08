# AUTOJUDGE: Programming Problem Difficulty Predictor
## Overview

Competitive programming platforms (Kattis, Codeforces, etc.) classify problems by difficulty.

This project automates that process using Machine Learning while using only **textual description**.

By analyzing the textual description of a problem, the system predicts:
**Difficulty Class**: Categorizes problems into Easy, Medium, or Hard
**Difficulty Score**: Assigns a precise numerical score (1 – 10)

## DataSet Used
The dataset was given dataset by ACM which contained:
[]Title
[]Description
[]input_description
[]output_description
[]sample_io
[]problem_class
[]problem_score
[]url


## Data Preprocessing 
1. The title,sample_io,url did not seem to contribute to guess the difficulty level so they were removed.
2. No null entries, but handled the empty strings by replacing.
3. Kept the numbers and math symbols for sure as they tell how tough a problem could be.
4. Converted the special characters like ö to normal characters.
5. Lowercased everything.
6. Removed stop words while preseving few.

## Feature Engineering

Simple word counts were insufficient.
The final model uses a Hybrid Feature Set to capture both technical language and mathematical constraints.

### 1️. NLP Features

1. TF-IDF Vectorization: Analyzes unigrams and bigrams
2. Logic-Aware Cleaning: Custom preprocessing preserves:
      1. Programming keywords (if, while, for)
      2. LaTeX-style mathematical symbols (\leq, \geq)
         (These are often removed by standard cleaners)

### 2. Handcrafted Features
1. Power: Detects constraints like 10^5 or 10^9 to estimate time complexity
2. Algorithm Density: Frequency of keywords like **dp, graph, union-find, recursion**
3. Math Symbol Count: Density of operators (+ - * /) and comparison symbols (< > =)
4. Constraint Complexity: Pattern detection such as N ≤ 10^5

## Model Selection 
### 1. Classification (Easy / Medium / Hard)

We compared four models to determine which best handled the nuances of "Medium" difficulty problems.


| Model               | Accuracy | F1-Score (Medium) | Observation                                           |
| ------------------- | -------- | ----------------- | ----------------------------------------------------- |
| Random Forest       | 51%      | 0.65              | Best overall accuracy; very strong on Medium problems |
| Naive Bayes         | 48%      | 0.65              | High recall for Medium, but ignores Hard problems     |
| Logistic Regression | 46%      | 0.56              | Most balanced; best at catching Hard problems         |
| SVM (Linear)        | 45%      | 0.56              | High precision but computationally slower             |


Confusion Matrices
| **Random Forest** | **Logistic Regression** | **Naive Bayes** | **SVM(Linear)** |
| ----------------- | ----------------------- | --------------- | --------------- |
| **E  M  H**       | **E  M  H**             | **E  M  H**     | **E  M  H**     |
| E: 46  102   5    | E: 65   73   15         | E: 12  141   0  | E:60 80 13      |
| M: 25  327  37    | M: 91  215   83         | M:  7  380   2  | M:20 214 155    |
| H: 22  217  42    | H: 20  160  101         | H: 12  263   6  | H:102 81 98     |




### 2️. Regression (Numerical Score)

Evaluated using:
MAE (Mean Absolute Error) → lower is better
R² Score → higher is better

| Model                   | MAE (↓ Better) | R² (↑ Better) |
| ----------------------- | -------------- | ------------- |
| Random Forest Regressor | 1.70           | 0.42          |
| Gradient Boosting       | 1.71           | 0.40          |
| Linear Regression       | 1.97           | 0.28          |


##  How to Run (VS Code Setup)
### 1️. Environment Setup

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

### 2️. Training the Model

Process the raw data and generate the saved model files:

```bash
python src/train.py
```

This creates the /models folder containing your .joblib files.

### 3️. Launching the App

Start the Streamlit web interface:

```bash
streamlit run app.py
```



