import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from scipy.sparse import hstack, csr_matrix
import os


from preprocessing import clean_text, extract_handcrafted_features

def main():
    # 1. Load Data
    print("Step 1: Loading raw data...")
    if not os.path.exists("data/problems_data.jsonl"):
        print("Error: data/problems_data.jsonl not found!")
        return
        
    df = pd.read_json("data/problems_data.jsonl", lines=True)

    # 2. Map and Filter
    class_mapping = {'easy': 0, 'medium': 1, 'hard': 2}
    df['problem_class_encoded'] = df['problem_class'].map(class_mapping)

    # Combine text fields for cleaning
    df['combined_text'] = df['description'].fillna('') + " " + \
                          df['input_description'].fillna('') + " " + \
                          df['output_description'].fillna('')

    # Drop rows with missing labels or empty text
    initial_count = len(df)
    df = df.dropna(subset=['problem_class_encoded', 'problem_score'])
    df = df[df['combined_text'].str.strip() != ""].copy()
    
    print(f"Step 2: Cleaned data. Kept {len(df)} rows out of {initial_count}.")

    # 3. Preprocessing
    print("Step 3: Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Select the required columns
    cleaned_df = df[['cleaned_text', 'problem_class_encoded', 'problem_score']]

    # Save to JSONL format
    # orient='records' and lines=True makes it JSONL
    save_path = "data/cleaned_problems_data.jsonl"
    cleaned_df.to_json(save_path, orient='records', lines=True)

    
    # 4. Feature Extraction
    print("Step 4: Vectorizing and extracting features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_text'])

    # The fix: use .values.tolist() to handle both Series and DataFrames
    handcrafted_list = df['combined_text'].apply(extract_handcrafted_features).values.tolist()
    X_handcrafted = csr_matrix(handcrafted_list)
    
    # Combine TF-IDF and Handcrafted features
    X_final = hstack([X_tfidf, X_handcrafted])

    y_class = df['problem_class_encoded'].astype(int)
    y_score = df['problem_score']

    # 5. Split Data
    # We split using the index so we can keep the class and score aligned
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X_final, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # 6. Train Models
    print("Step 5: Training Models...")
    
    # Classification
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train_class)

    # Regression 
    # (Fix: We use .loc to match the training indices to the scores)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_score.loc[y_train_class.index])

    # 7. Evaluate
    print("\n--- Classification Report ---")
    y_pred_class = clf.predict(X_test)
    print(classification_report(y_test_class, y_pred_class, target_names=['Easy', 'Medium', 'Hard']))

    print("\n--- Regression Metrics ---")
    y_pred_score = reg.predict(X_test)
    print(f"Mean Absolute Error: {mean_absolute_error(y_score.loc[y_test_class.index], y_pred_score):.2f}")
    print(f"R2 Score: {r2_score(y_score.loc[y_test_class.index], y_pred_score):.2f}")

    # 8. Save Assets
    print("\nStep 6: Saving models to /models folder...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/classifier.joblib')
    joblib.dump(reg, 'models/regressor.joblib')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
    
    print("Success! All models saved.")

if __name__ == "__main__":
    main()