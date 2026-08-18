

import os
from datasets import load_dataset
from typing import Dict, List

class TextDataLoader:
    """
    Responsible for loading the IMDb and Rotten Tomatoes datasets,
    and preparing the combined text corpus for Word2Vec training.
    """
    
    def __init__(self):
        # We will store the loaded huggingface datasets here
        self.datasets: Dict[str, any] = {}
        
    def load_classification_datasets(self):
        """
        Loads the standard text classification datasets defined in the project.
        """
        # Load IMDb dataset
        print("Loading IMDb dataset...")
        self.datasets['imdb'] = load_dataset("imdb")
        
        # Load Rotten Tomatoes dataset
        print("Loading Rotten Tomatoes dataset...")
        self.datasets['rotten_tomatoes'] = load_dataset("rotten_tomatoes")
        
        return self.datasets

    def extract_all_texts(self) -> List[str]:
        """
        Extracts raw texts from all available splits of the loaded datasets.
        This is useful for creating a massive corpus for embedding training.
        """
        if not self.datasets:
            self.load_classification_datasets()
            
        all_texts = []
        
        for dataset_name, dataset_dict in self.datasets.items():
            for split in dataset_dict.keys(): # train, test, unsupervised, validation
                # Extract the 'text' column from the current split
                texts = dataset_dict[split]['text']
                all_texts.extend(texts)
                
        return all_texts

    def create_combined_corpus_file(self, output_filepath: str):
        """
        Combines a few sources of data into a single text file,
        as requested by the project instructions for Word2Vec training.
        """
        print(f"Extracting all texts to create combined corpus at {output_filepath}...")
        all_texts = self.extract_all_texts()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for text in all_texts:
                # Remove newlines within the text to keep one document per line
                clean_text = text.replace('\n', ' ').replace('\r', '')
                f.write(clean_text + '\n')
                
        print(f"Successfully saved {len(all_texts)} documents to {output_filepath}")

# Quick test/usage example (can be removed later or moved to main)
if __name__ == "__main__":
    loader = TextDataLoader()
    # Saves the combined texts into the data folder
    loader.create_combined_corpus_file("../data/combined_corpus.txt")