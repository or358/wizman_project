
import os
import logging
from gensim.models import Word2Vec
from data_loader import TextDataLoader
from preprocessor import TextPreprocessor
from config import Word2VecConfig

# Enable Gensim's internal logging to see progress bar, ETA, and word counts in terminal
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class CorpusIterator:
    """
    An iterator that yields one preprocessed document at a time.
    Prevents RAM overload by reading data row-by-row (Streaming).
    """
    def __init__(self, data_loader, preprocessor):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        
        if not self.data_loader.datasets:
            self.data_loader.load_classification_datasets()

    def __iter__(self):
        for dataset_name, dataset_dict in self.data_loader.datasets.items():
            for split in dataset_dict.keys():
                for example in dataset_dict[split]:
                    # Extract text, normalize and tokenize on the fly
                    tokenized_text = self.preprocessor.tokenize(example['text'])
                    if tokenized_text:
                        yield tokenized_text

def train_and_save_word2vec(output_model_path: str = "../data/my_word2vec.model"):
    """
    Trains a Gensim Word2Vec model using a memory-friendly streaming approach.
    """
    loader = TextDataLoader()
    preprocessor = TextPreprocessor()
    
    corpus_iterator = CorpusIterator(loader, preprocessor)
    
    print("Initializing Word2Vec model...")
    
    # Initialize model without sentences to allow manual step-by-step processing
    model = Word2Vec(
        vector_size=Word2VecConfig.VECTOR_SIZE,
        window=Word2VecConfig.WINDOW,
        min_count=Word2VecConfig.MIN_COUNT,
        sg=Word2VecConfig.SG,
        negative=Word2VecConfig.NEGATIVE,
        ns_exponent=Word2VecConfig.NS_EXPONENT,
        workers=3
    )

    # Step 1: Build Vocabulary (This counts the words)
    print("Step 1/2: Building vocabulary and counting words...")
    model.build_vocab(corpus_iterator)
    
    # Print Statistics
    total_words = model.corpus_total_words
    vocab_size = len(model.wv.index_to_key)
    print("-" * 30)
    print(f"Corpus Statistics:")
    print(f" - Total tokens processed: {total_words:,}")
    print(f" - Unique tokens in vocabulary (after min_count={Word2VecConfig.MIN_COUNT}): {vocab_size:,}")
    print("-" * 30)

    # Step 2: Training
    print(f"Step 2/2: Training Word2Vec model for {Word2VecConfig.EPOCHS} epochs...")
    model.train(
        corpus_iterable=corpus_iterator,
        total_examples=model.corpus_count,
        epochs=Word2VecConfig.EPOCHS
    )
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    print(f"Saving trained Word2Vec model to {output_model_path}...")
    model.save(output_model_path)
    print("Word2Vec training completed successfully!")

if __name__ == "__main__":
    train_and_save_word2vec()