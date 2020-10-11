## Learning to smell

import numpy as np
import pandas as pd

# read dataset

train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
vocab = open("data/vocabulary.txt", 'r').read().split("\n")

print(f"size of training {train_set.shape}")
print(f"size of testing {test_set.shape}")
# print(type(vocab))

# Prepare training set
# Generate image

# Prepare test set
def onehot_sentence(sentence):
    l = np.zeros(len(vocab))
    for label in sentence.split(','):
        l[vocab.index(label)] = 1
    return l

test_vocab = np.zeros((train_set.shape[0], len(vocab)), dtype=np.float32)

for i in range(train_set.shape[0]):
    test_vocab[i] = onehot_sentence(train_set.SENTENCE.iloc[i])

# shuffle index
# np.random.shuffle(train_set)
# np.random.shuffle(test_set)

# split training set into actual training set and validation set
from sklearn.model_selection import train_test_split as splitter
x_train, x_val, y_train, y_val = splitter(train_set, test_vocab, test_size=0.2, random_state=0)

# Build model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.losses import BinaryCrossentropy

# Model = Sequential()
