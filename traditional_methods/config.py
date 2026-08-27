"""
Configuration file for the Traditional Methods Sentiment Analysis project.
Centralizes all hyperparameters and global settings.
"""

# Data Settings
# Switch between "stanfordnlp/imdb" and "cornell-movie-review-data/rotten_tomatoes" as needed
DATASET_NAME = "stanfordnlp/imdb"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing Settings
MAX_FEATURES = 5000

# Model Hyperparameters
LR_C = 1.0
LR_MAX_ITER = 1000

SVM_C = 1.0

DT_MAX_DEPTH = 20