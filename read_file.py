## Learning to smell

import numpy as np
import pandas as pd

# read dataset

training = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")
vocab = open("vocabulary.txt", 'r').read().split("\n")

print(f"size of training {training.shape}")
print(f"size of testing {test.shape}")
# print(type(vocab))

# Prepare training set
# Generate image

# Prepare test set
def onehot_sentence(sentence):
    l = np.zeros(len(vocab))
    for label in sentence.split(','):
        l[vocab.index(label)] = 1
    return l

test_vocab = np.zeros((training.shape[0], len(vocab)), dtype=np.float32)

for i in range(training.shape[0]):
    test_vocab[i] = onehot_sentence(training.SENTENCE.iloc[i])

# shuffle index
np.random.shuffle(training)
np.random.shuffle(test)

# split training set into actual training set and validation set
from sklearn.model_selection import train_test_split as splitter
x_train, x_val, y_train, y_val = splitter(training, test_vocab, test_size=0.2, random_state=0)

# Build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

Model = Sequential()
