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

# Get a flattened list of all labels
labels = [label.split(',') for label in train_set.SENTENCE.tolist()]
labels = [item for sublist in labels for item in sublist]

# Count the number of occurances for each label
counts = dict((x, labels.count(x)) for x in set(labels))
pprint(counts)