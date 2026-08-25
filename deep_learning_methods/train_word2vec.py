# deep_learning_methods/train_word2vec.py

import os
import logging
from gensim.models import Word2Vec
from data_loader import TextDataLoader
from preprocessor import TextPreprocessor
from config import Word2VecConfig

# Enable Gensim's internal logging to see progress bar, ETA, and word counts in terminal
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_all_sentences_to_ram(data_loader, preprocessor):
    """
    Loads and pre-processes all texts into RAM.
    This is extremely fast if the server has enough memory, 
    as it avoids repeated tokenization during Gensim's multiple epochs.
    """
    if not data_loader.datasets:
        data_loader.load_classification_datasets()

    print("Loading and tokenizing all datasets into RAM. Please wait...")
    all_sentences = []
    
    for dataset_name, dataset_dict in data_loader.datasets.items():
        for split in dataset_dict.keys():
            for example in dataset_dict[split]:
                # Extract text, normalize and tokenize ONCE
                tokenized_text = preprocessor.tokenize(example['text'])
                if tokenized_text:
                    all_sentences.append(tokenized_text)
                    
    print(f"Successfully loaded {len(all_sentences):,} sentences into memory.")
    return all_sentences

def train_and_save_word2vec(output_model_path: str = "/data/my_word2vec.model"):
    """
    Trains a Gensim Word2Vec model using an in-memory approach for maximum speed.
    """
    loader = TextDataLoader()
    preprocessor = TextPreprocessor()
    
    # Load everything to RAM once!
    sentences_list = load_all_sentences_to_ram(loader, preprocessor)
    
    print("Initializing and training Word2Vec model...")
    
    # Pass the fully processed list directly to Word2Vec.
    # Gensim automatically builds the vocabulary and trains all epochs.
    model = Word2Vec(
        sentences=sentences_list,
        vector_size=Word2VecConfig.VECTOR_SIZE,
        window=Word2VecConfig.WINDOW,
        min_count=Word2VecConfig.MIN_COUNT,
        sg=Word2VecConfig.SG,
        negative=Word2VecConfig.NEGATIVE,
        ns_exponent=Word2VecConfig.NS_EXPONENT,
        epochs=Word2VecConfig.EPOCHS,
        workers=8  # Using 8 CPU threads for maximum parallel processing
    )
    
    # Print Statistics
    total_words = model.corpus_total_words
    vocab_size = len(model.wv.index_to_key)
    print("-" * 30)
    print(f"Corpus Statistics:")
    print(f" - Total tokens processed: {total_words:,}")
    print(f" - Unique tokens in vocabulary (after min_count={Word2VecConfig.MIN_COUNT}): {vocab_size:,}")
    print("-" * 30)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    print(f"Saving trained Word2Vec model to {output_model_path}...")
    model.save(output_model_path)
    print("Word2Vec training completed successfully!")

if __name__ == "__main__":
    train_and_save_word2vec()