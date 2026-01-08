import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix

# We import the functions we just wrote in preprocessing.py
from preprocessing import clean_text, extract_handcrafted_features

def main():
    # 1. Load Data
    # Ensure your problems_data.jsonl is inside a folder named 'data'
    print("Step 1: Loading raw data...")
    df = pd.read_json("data/problems_data.jsonl", lines=True)

    # 2. Preprocessing
    print("Step 2: Cleaning text and extracting features...")
    # Combine all text fields into one
    df['combined_text']=  df['description'].fillna('') + " " + \
                          df['input_description'].fillna('') + " " + \
                          df['output_description'].fillna('')
    
    # Clean the text using our modular function
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    # Extract numeric handcrafted features
    handcrafted_features = df['combined_text'].apply(extract_handcrafted_features)
    
    # 3. TF-IDF Vectorization
    print("Step 3: Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['cleaned_text'])

    # 4. Combine Features
    # This creates the same 'X_final' we used in the notebook
    X_final = hstack([X_tfidf, csr_matrix(handcrafted_features.values)])

    # 5. Prepare Targets
    class_mapping = {'Easy': 0, 'Medium': 1, 'Hard': 2}
    y_class = df['problem_class'].map(class_mapping)
    y_score = df['problem_score']

    # 6. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X_final, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # 7. Train Models
    print("Step 4: Training Random Forest Classifier & Regressor...")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train_class)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    # We use the same training indices to align the scores
    reg.fit(X_train, y_score.iloc[y_train_class.index])

    # 8. Honest Evaluation
    print("\n--- Final Performance Evaluation ---")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test_class, y_pred, target_names=['Easy', 'Medium', 'Hard']))

    # 9. Save Artifacts
    print("\nStep 5: Saving models for deployment...")
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # We save three things: The Classifier, The Regressor, and the TF-IDF Vectorizer
    joblib.dump(clf, 'models/classifier.joblib')
    joblib.dump(reg, 'models/regressor.joblib')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
    
    print("Success! Models are saved in the /models folder.")

if __name__ == "__main__":
    main()