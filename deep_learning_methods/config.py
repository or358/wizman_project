

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
    General configuration for the deep learning models (CNN/LSTM).
    """
    UNK_TOKEN = "<UNK>"    # Token used for missing or rare words
    PAD_TOKEN = "<PAD>"    # Token used for padding sequences to the same length