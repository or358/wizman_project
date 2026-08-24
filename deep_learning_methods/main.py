# deep_learning_methods/main.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from gensim.models import Word2Vec
import torchtext.vocab as vocab_lib

from config import ModelConfig, Word2VecConfig
from preprocessor import TextPreprocessor
from data_loader import Vocabulary, TextDataset
from models import TextCNN, TextLSTM
from trainer import ModelTrainer

def load_pretrained_glove(vocab: Vocabulary):
    """
    Loads pretrained GloVe embeddings for the words in our vocabulary.
    Returns an embedding matrix matching our vocabulary size.
    """
    print("\nDownloading/Loading GloVe pretrained vectors (this might take a minute the first time)...")
    # Using GloVe 6B tokens, 300 dimensions as a standard pretrained model
    glove = vocab_lib.GloVe(name='6B', dim=300)
    
    embedding_matrix = torch.zeros((vocab.vocab_size, 300))
    nn.init.uniform_(embedding_matrix, -0.1, 0.1)
    embedding_matrix[vocab.word2idx[ModelConfig.PAD_TOKEN]] = torch.zeros(300)
    
    words_found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove.stoi:
            embedding_matrix[idx] = glove.vectors[glove.stoi[word]]
            words_found += 1
            
    print(f"Loaded {words_found}/{vocab.vocab_size} pretrained GloVe vectors.")
    return embedding_matrix, 300 # Return matrix and the dimension size

def create_our_w2v_matrix(vocab: Vocabulary, w2v_path: str = "../data/my_word2vec.model"):
    """
    Loads our custom trained Word2Vec model.
    """
    if not os.path.exists(w2v_path):
        raise FileNotFoundError(f"Cannot find Word2Vec model at {w2v_path}. Did you run train_word2vec.py?")
        
    print(f"\nLoading our custom trained Word2Vec model from {w2v_path}...")
    w2v_model = Word2Vec.load(w2v_path)
    
    embedding_matrix = torch.zeros((vocab.vocab_size, Word2VecConfig.VECTOR_SIZE))
    nn.init.uniform_(embedding_matrix, -0.1, 0.1)
    embedding_matrix[vocab.word2idx[ModelConfig.PAD_TOKEN]] = torch.zeros(Word2VecConfig.VECTOR_SIZE)
    
    words_found = 0
    for word, idx in vocab.word2idx.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = torch.tensor(w2v_model.wv[word])
            words_found += 1
            
    print(f"Loaded {words_found}/{vocab.vocab_size} vectors from our custom Word2Vec.")
    return embedding_matrix, Word2VecConfig.VECTOR_SIZE

def prepare_data_for_dataset(dataset_name: str, preprocessor: TextPreprocessor):
    """Loads dataset, builds vocabulary, and returns DataLoaders."""
    print(f"\nLoading and preparing dataset: {dataset_name.upper()}...")
    
    if dataset_name == 'rotten_tomatoes':
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
        val_texts, val_labels = dataset['validation']['text'], dataset['validation']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
        
    elif dataset_name == 'imdb':
        dataset = load_dataset("stanfordnlp/imdb")
        full_train = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_texts, train_labels = full_train['train']['text'], full_train['train']['label']
        val_texts, val_labels = full_train['test']['text'], full_train['test']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
    
    print("Tokenizing training set to build vocabulary...")
    tokenized_train = [preprocessor.tokenize(text) for text in train_texts]
    
    vocab = Vocabulary()
    vocab.build_vocab(tokenized_train, min_count=2)
    print(f"Vocabulary size created: {vocab.vocab_size}")
    
    train_dataset = TextDataset(train_texts, train_labels, vocab, preprocessor)
    val_dataset = TextDataset(val_texts, val_labels, vocab, preprocessor)
    test_dataset = TextDataset(test_texts, test_labels, vocab, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=False)
    
    return vocab, train_loader, val_loader, test_loader

def run_interactive_experiment():
    """Runs an interactive CLI to configure and launch an experiment."""
    print("="*60)
    print(" DEEP LEARNING CLASSIFICATION - INTERACTIVE EXPERIMENT")
    print("="*60)
    
    # 1. Choose Dataset
    while True:
        ds_choice = input("Select Dataset (1 for IMDb, 2 for Rotten Tomatoes): ").strip()
        if ds_choice in ['1', '2']:
            break
    dataset_name = 'imdb' if ds_choice == '1' else 'rotten_tomatoes'
    
    # 2. Choose Architecture
    while True:
        model_choice = input("Select Model Architecture (1 for CNN, 2 for LSTM): ").strip()
        if model_choice in ['1', '2']:
            break
    model_type = 'CNN' if model_choice == '1' else 'LSTM'
    
    # 3. Choose Embedding Method
    print("\nEmbedding Methods:")
    print("1. Random Generation")
    print("2. Pretrained Models (GloVe)")
    print("3. Our Trained Word2Vec")
    while True:
        emb_choice = input("Select Embedding Method (1, 2, or 3): ").strip()
        if emb_choice in ['1', '2', '3']:
            break
            
    # 4. Choose Freezing Option
    while True:
        freeze_choice = input("Freeze embeddings during training? (y/n): ").strip().lower()
        if freeze_choice in ['y', 'n']:
            break
    freeze_embedding = (freeze_choice == 'y')
    
    print("\n" + "-"*40)
    print("EXPERIMENT CONFIGURATION SUMMARY:")
    print(f"Dataset:       {dataset_name.upper()}")
    print(f"Model:         {model_type}")
    print(f"Embedding:     {'Random' if emb_choice=='1' else 'GloVe Pretrained' if emb_choice=='2' else 'Custom Word2Vec'}")
    print(f"Freeze Emb:    {freeze_embedding}")
    print("-"*40)
    
    # Start actual execution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing process on device: {device}")
    
    preprocessor = TextPreprocessor()
    vocab, train_loader, val_loader, test_loader = prepare_data_for_dataset(dataset_name, preprocessor)
    
    # Prepare embeddings based on choice
    embedding_matrix = None
    embedding_dim = Word2VecConfig.VECTOR_SIZE # Default
    
    if emb_choice == '1':
        print("\nUsing Randomly Generated Embeddings.")
        embedding_matrix = None # Handled inside the model creation
        
    elif emb_choice == '2':
        embedding_matrix, embedding_dim = load_pretrained_glove(vocab)
        
    elif emb_choice == '3':
        embedding_matrix, embedding_dim = create_our_w2v_matrix(vocab)
    
    # We need to temporarily update Word2VecConfig so the model knows the right dimension (e.g. GloVe is 300)
    original_vector_size = Word2VecConfig.VECTOR_SIZE
    Word2VecConfig.VECTOR_SIZE = embedding_dim
    
    print(f"\nBuilding {model_type} Model...")
    if model_type == 'CNN':
        model = TextCNN(vocab.vocab_size, embedding_matrix, freeze_embedding)
    else:
        model = TextLSTM(vocab.vocab_size, embedding_matrix, freeze_embedding)
        
    # Revert config to original just in case
    Word2VecConfig.VECTOR_SIZE = original_vector_size
        
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=ModelConfig.LEARNING_RATE,
        device=device,
        experiment_name=f"{dataset_name}_{model_type}_Freeze{freeze_embedding}" 
    )
    
    # Run training for 5 epochs (you can adjust this)
    trainer.train(num_epochs=5)
    
    print("\nExperiment finished. You can run the script again to test another configuration.")

if __name__ == "__main__":
    # Ensure torchtext is installed for GloVe
    try:
        import torchtext
    except ImportError:
        print("Error: 'torchtext' is required for pretrained GloVe embeddings.")
        print("Please install it using: pip install torchtext")
        exit(1)
        
    run_interactive_experiment()