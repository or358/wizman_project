import pandas as pd
from datasets import load_dataset

def load_data():
    """
    Downloads and loads the IMDB dataset via Hugging Face datasets.
    Returns a unified pandas DataFrame containing both train and test splits.
    """
    print("Downloading/Loading IMDB dataset via Hugging Face...")
    dataset = load_dataset("stanfordnlp/imdb")
    
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    
    # Combine train and test sets into a single DataFrame
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    print(f"Data loaded successfully. Total reviews: {len(df)}")
    return df

if __name__ == "__main__":
    dataset = load_data()
    print(dataset.head())