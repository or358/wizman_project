

import torch
import torch.nn as nn
from typing import Optional
from config import CNNConfig, LSTMConfig, ModelConfig, Word2VecConfig

def create_embedding_layer(vocab_size: int, embedding_matrix: Optional[torch.Tensor], freeze: bool) -> nn.Embedding:
    """
    Creates an embedding layer for the models.
    Supports either random initialization or pre-trained weights (like our Word2Vec).
    The 'freeze' parameter determines if the weights are updated during training.
    """
    if embedding_matrix is not None:
        # Load the pre-trained Word2Vec weights
        embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=freeze, padding_idx=0)
    else:
        # Create random weights (one of the project requirements)
        embedding = nn.Embedding(vocab_size, Word2VecConfig.VECTOR_SIZE, padding_idx=0)
        # If random, it must be trainable (freeze=False makes no sense for random initialization)
        embedding.weight.requires_grad = not freeze
        
    return embedding


class TextCNN(nn.Module):
    """
    CNN for text classification.
    Uses multiple filter sizes to capture different n-gram patterns.
    """
    def __init__(self, vocab_size: int, embedding_matrix: Optional[torch.Tensor] = None, freeze_embedding: bool = False):
        super(TextCNN, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = create_embedding_layer(vocab_size, embedding_matrix, freeze_embedding)
        
        # 2. Convolutional Layers (one for each filter size)
        # ModuleList acts like a standard Python list but tells PyTorch about these layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, 
                      out_channels=CNNConfig.NUM_FILTERS, 
                      kernel_size=(fs, Word2VecConfig.VECTOR_SIZE)) 
            for fs in CNNConfig.FILTER_SIZES
        ])
        
        # 3. Dropout (prevents overfitting)
        self.dropout = nn.Dropout(CNNConfig.DROPOUT)
        
        # 4. Fully Connected (Linear) Layer for final classification
        # The input size is num_filters * number of different filter sizes
        self.fc = nn.Linear(len(CNNConfig.FILTER_SIZES) * CNNConfig.NUM_FILTERS, ModelConfig.NUM_CLASSES)
        
    def forward(self, x):
        # x shape: [batch_size, max_seq_length]
        
        # Add an extra dimension for the 'channel' (like color channels in images)
        embedded = self.embedding(x).unsqueeze(1) 
        # embedded shape: [batch_size, 1, max_seq_length, embedding_dim]
        
        # Apply convolution -> activation function (ReLU) -> max pooling
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate the pooled features
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # Final classification
        return self.fc(cat)


class TextLSTM(nn.Module):
    """
    LSTM for text classification.
    Reads the sequence of words and captures long-term dependencies.
    """
    def __init__(self, vocab_size: int, embedding_matrix: Optional[torch.Tensor] = None, freeze_embedding: bool = False):
        super(TextLSTM, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = create_embedding_layer(vocab_size, embedding_matrix, freeze_embedding)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(input_size=Word2VecConfig.VECTOR_SIZE,
                            hidden_size=LSTMConfig.HIDDEN_DIM,
                            num_layers=LSTMConfig.NUM_LAYERS,
                            bidirectional=LSTMConfig.BIDIRECTIONAL,
                            dropout=LSTMConfig.DROPOUT if LSTMConfig.NUM_LAYERS > 1 else 0,
                            batch_first=True)
        
        # 3. Dropout
        self.dropout = nn.Dropout(LSTMConfig.DROPOUT)
        
        # 4. Fully Connected Layer
        # If bidirectional, the hidden state size is doubled
        hidden_factor = 2 if LSTMConfig.BIDIRECTIONAL else 1
        self.fc = nn.Linear(LSTMConfig.HIDDEN_DIM * hidden_factor, ModelConfig.NUM_CLASSES)
        
    def forward(self, x):
        # x shape: [batch_size, max_seq_length]
        
        embedded = self.dropout(self.embedding(x))
        # embedded shape: [batch_size, max_seq_length, embedding_dim]
        
        # Output contains the hidden states for each time step
        # Hidden contains the final hidden state
        output, (hidden, cell) = self.lstm(embedded)
        
        # We use the final hidden state for classification
        if LSTMConfig.BIDIRECTIONAL:
            # Concat the final forward and backward hidden states
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)