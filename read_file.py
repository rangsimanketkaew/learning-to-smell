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
