import numpy as np


def onehot_sentence(vocab, sentence):
    l = np.zeros(len(vocab))
    for label in sentence:
        l[vocab.index(label)] = 1
    return l
