import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Removes HTML tags and non-alphabetic characters, and converts text to lowercase.
    """
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

def preprocess_and_split(df):
    """
    Applies text cleaning, TF-IDF vectorization, and train-test splitting.
    """
    print("1. Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    print("2. Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    print("3. Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Done! Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, vectorizer