# deep_learning_methods/data_loader.py

import torch
from datasets import load_dataset
from typing import Dict, List
from torch.utils.data import Dataset
from config import ModelConfig

class TextDataLoader:
    """
    Responsible for loading the standard datasets and managing the raw text.
    """
    
    def __init__(self):
        self.datasets: Dict[str, any] = {}
        
    def load_classification_datasets(self):
        """
        Loads the datasets and controls the exact size to reach ~60 Million words.
        """
        print("Loading IMDb dataset...")
        self.datasets['imdb'] = load_dataset("stanfordnlp/imdb")
        
        print("Loading Rotten Tomatoes dataset...")
        self.datasets['rotten_tomatoes'] = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        
        print("Loading 60,000 Yelp reviews to guarantee ~40M total words...")
        self.datasets['yelp'] = {
            "train": load_dataset("fancyzhx/yelp_polarity", split="train[:60000]")
        }
        
        return self.datasets


class Vocabulary:
    """
    Manages the mapping between words (strings) and indices (integers).
    """
    def __init__(self):
        self.word2idx: Dict[str, int] = {
            ModelConfig.PAD_TOKEN: 0,
            ModelConfig.UNK_TOKEN: 1
        }
        self.idx2word: Dict[int, str] = {
            0: ModelConfig.PAD_TOKEN,
            1: ModelConfig.UNK_TOKEN
        }
        self.vocab_size = 2
        
    def build_vocab(self, tokenized_corpus: List[List[str]], min_count: int = 3):
        """Builds vocabulary, ignoring words with frequency < min_count."""
        word_freq = {}
        for doc in tokenized_corpus:
            for word in doc:
                word_freq[word] = word_freq.get(word, 0) + 1
                
        for word, freq in word_freq.items():
            if freq >= min_count and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def text_to_indices(self, tokenized_text: List[str]) -> List[int]:
        unk_idx = self.word2idx[ModelConfig.UNK_TOKEN]
        return [self.word2idx.get(word, unk_idx) for word in tokenized_text]


class TextDataset(Dataset):
    """
    A PyTorch Dataset that handles padding and returns (features, label).
    """
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, preprocessor):
        self.labels = labels
        self.max_length = ModelConfig.MAX_SEQ_LENGTH
        self.pad_idx = vocab.word2idx[ModelConfig.PAD_TOKEN]
        
        self.sequences = []
        for text in texts:
            tokens = preprocessor.tokenize(text)
            indices = vocab.text_to_indices(tokens)
            self.sequences.append(indices)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Truncate
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            
        # Pad
        pad_length = self.max_length - len(sequence)
        sequence = sequence + [self.pad_idx] * pad_length
            
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float32)