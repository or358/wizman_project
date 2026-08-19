

class Word2VecConfig:
    """
    Configuration parameters for training the Gensim Word2Vec model.
    These parameters follow the specific requirements of the project.
    """
    VECTOR_SIZE = 512      # Dimensionality of the word vectors
    WINDOW = 5             # Maximum distance between the current and predicted word
    MIN_COUNT = 3          # Ignores all words with total frequency lower than this
    SG = 1                 # 1 for skip-gram architecture, 0 for CBOW
    NEGATIVE = 20          # Number of negative samples to be drawn
    NS_EXPONENT = 0.75     # Exponent used to shape the negative sampling distribution
    EPOCHS = 7             # Number of iterations (epochs) over the corpus

class ModelConfig:
    """
    General configuration for the deep learning models and data padding.
    """
    UNK_TOKEN = "<UNK>"    
    PAD_TOKEN = "<PAD>"    
    MAX_SEQ_LENGTH = 200   # Maximum number of words in a review (pad/truncate to this)
    BATCH_SIZE = 32        # Number of samples passed to the network at once
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2        # Positive (1) or Negative (0) review

class CNNConfig:
    """Configuration specific to the CNN model."""
    NUM_FILTERS = 100            # Number of filters per window size
    FILTER_SIZES = [3, 4, 5]     # Window sizes (like 3-grams, 4-grams, 5-grams)
    DROPOUT = 0.5

class LSTMConfig:
    """Configuration specific to the LSTM model."""
    HIDDEN_DIM = 256             # Number of hidden units in the LSTM
    NUM_LAYERS = 2               # Number of stacked LSTM layers
    BIDIRECTIONAL = True         # Should the LSTM read text forwards and backwards?
    DROPOUT = 0.5