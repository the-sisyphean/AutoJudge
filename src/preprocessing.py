import re
import string
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'if', 'while', 'for', 'each'}

def clean_text(text):
    """Refined cleaning logic from your final notebook."""
    if not isinstance(text, str): return ""
    
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").lower()
    
    # 1. Preserve Algo Keywords
    algo_keywords = {
        r'union[ \-]find': ' union_find ',
        r'dynamic[ \-]programming': ' dynamic_programming ',
        r'segment[ \-]trees?': ' segment_tree ',
        r'binary[ \-]search': ' binary_search ',
        r'shortest[ \-]path': ' shortest_path '
    }
    for pattern, replacement in algo_keywords.items():
        text = re.sub(pattern, replacement, text)

    # 2. LaTeX & Spaced Numbers
    replacements = {r'\\leq': ' leq ', r'\\geq': ' geq ', r'\\neq': ' neq ', r'\\cdot': ' times '}
    for latex, word in replacements.items():
        text = re.sub(latex, word, text)
    text = re.sub(r'(\d+)([,\\\s]+)(\d+)', r'\1\3', text)

    # 3. Standard Cleaning
    text = re.sub(r"[()\n!,:'\"$.?{}\\/]", " ", text)
    text = re.sub(r'\d+', ' num ', text)
    
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1])

def extract_max_power(text):
    """Extracts the highest 10^X constraint."""
    powers = re.findall(r'10\^\{?(\d+)\}?', str(text))
    return max([int(p) for p in powers]) if powers else 0

def extract_handcrafted_features(text):
    """Calculates algorithm counts, math terms, and constraint patterns."""
    text_lower = str(text).lower()
    
    algos = ['union find', 'dynamic programming', 'segment tree', 'graph', 'shortest path']
    math_terms = ['modulo', 'prime', 'probability', 'matrix', 'gcd']
    
    features = {
        'max_10_power': extract_max_power(text),
        'algo_count': sum(1 for a in algos if a in text_lower),
        'math_count': sum(1 for m in math_terms if m in text_lower),
        'word_count': len(text_lower.split()),
        'constraint_count': len(re.findall(r'[n_m_k_t]\s*[\\leq|\\le|<=|<]', text_lower))
    }
    return pd.Series(features)