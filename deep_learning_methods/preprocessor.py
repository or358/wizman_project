

import re
from typing import List
from config import ModelConfig

class TextPreprocessor:
    """
    Handles text normalization and preparation for deep learning models
    in a pre-transformers environment (word-level).
    """
    
    def __init__(self):
        self.unk_token = ModelConfig.UNK_TOKEN
    
    def normalize_text(self, text: str) -> str:
        """
        Normalizes a single text string by converting to lowercase 
        and separating punctuation marks with spaces.
        """
        # Convert text to lower case
        text = text.lower()
        
        # Separate punctuation signs from text using regex
        # The regex captures any character that is not a word character (\w) or whitespace (\s)
        # and adds a space before and after it.
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        
        # Remove multiple consecutive spaces that might have been created
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Splits the normalized text into a list of words (tokens).
        """
        normalized_text = self.normalize_text(text)
        # Split by space to get individual words/punctuation
        return normalized_text.split()

    def process_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Processes a full list of documents (corpus) and returns 
        a list of tokenized documents.
        """
        return [self.tokenize(doc) for doc in corpus]