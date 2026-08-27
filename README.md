# Sentiment Analysis Benchmarking Project

This project provides a comprehensive comparison of various machine learning and deep learning approaches for sentiment analysis. It benchmarks traditional models, classical deep learning architectures (CNN, LSTM) with various embedding strategies, and state-of-the-art Transformer-based models (DistilBERT).

## Project Structure

```text
.
├── traditional_methods/      # Traditional ML models (Logistic Regression, SVM, etc.)
│   ├── main.py               # Entry point for traditional methods
│   ├── models.py             # Model definitions
│   └── ...
├── deep_learning_methods/    # Deep Learning models (CNN, LSTM, Transformers)
│   ├── main.py               # Main experiment runner for DL models
│   ├── transformer_main.py   # Fine-tuning DistilBERT
│   ├── train_word2vec.py     # Script to train custom Word2Vec embeddings
│   ├── models.py             # CNN and LSTM architectures
│   └── ...
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Traditional Methods
To run the traditional machine learning experiments (Logistic Regression, Linear SVM, Decision Tree) on the default dataset:
```bash
python traditional_methods/main.py
```
Results and confusion matrices will be saved in the `traditional_results/` directory.

### 2. Deep Learning Methods (CNN & LSTM)
The DL experiments compare CNN and LSTM models across different datasets (IMDB, Rotten Tomatoes) and embedding strategies (Random, GloVe, Word2Vec).

**Note**: To use custom Word2Vec embeddings, you must first train the model:
```bash
python deep_learning_methods/train_word2vec.py
```

To run the full suite of classical DL experiments (24 combinations) + Transformer fine-tuning:
```bash
python deep_learning_methods/main.py
```

### 3. Transformer Experiments
To run only the DistilBERT fine-tuning experiments:
```bash
python deep_learning_methods/transformer_main.py
```

## Models & Methods

### Traditional Models
- **Logistic Regression** (with TF-IDF)
- **Linear SVM** (with TF-IDF)
- **Decision Tree** (with TF-IDF)

### Deep Learning Models
- **TextCNN**: Convolutional Neural Network for Text.
- **TextLSTM**: Long Short-Term Memory Network.
- **Embedding Strategies**:
  - Randomly Initialized
  - Pretrained **GloVe** (6B tokens, 300d)
  - Custom trained **Word2Vec**
- **Embedding Freezing**: Each model is tested with both frozen and trainable embeddings.

### Transformer Models
- **DistilBERT**: Fine-tuned `distilbert-base-uncased` from Hugging Face Transformers.

## Datasets
- **IMDB**: A large dataset of 50,000 movie reviews for binary sentiment classification.
- **Rotten Tomatoes**: A dataset containing movie reviews from the Rotten Tomatoes website.

## Results
- **Traditional Results**: Located in `traditional_results/`.
- **Deep Learning & Transformer Results**: Located in the `results/` directory, including:
  - CSV files with per-epoch metrics.
  - Loss and Accuracy graphs.
  - `master_summary_results.csv` containing summarized performance across all experiments.
