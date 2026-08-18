import pandas as pd
from datasets import load_dataset
import config

def load_data():
    """
    Downloads and loads the specified sentiment analysis dataset via Hugging Face.
    Handles differences in column naming between IMDb and Rotten Tomatoes.
    """
    print(f"Downloading/Loading {config.DATASET_NAME} dataset...")
    dataset = load_dataset(config.DATASET_NAME)
    
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Standardize column names based on the dataset structure
    if 'text' not in df.columns and 'review' in df.columns:
        df['text'] = df['review']
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
    print(f"Data loaded successfully. Total samples: {len(df)}")
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())