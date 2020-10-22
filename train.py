# Learning to smell

import numpy as np
import pandas as pd
import scipy.sparse as sp

# import deepchem as dc
# RDkit for fingerprinting and cheminformatics
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.Chem.EState import Fingerprinter
# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split
# TensorFlow and Keras for deep learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ReduceLROnPlateau
# from keras.regularizers import WeightRegularizer
from keras.optimizers import SGD


###############
# Read dataset
###############
whole_train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")
vocab = open("data/vocabulary.txt", 'r').read().split("\n")
sample_sub = pd.read_csv("data/sample_submission.csv")

###############
# Split dataset
###############

train_set, train_label = list(whole_train_set['SMILES']), list(whole_train_set['SENTENCE'])

train_set, valid_set, train_label, valid_label = train_test_split(
    train_set, train_label, test_size=0.2, random_state=42)

test_set = list(test_set['SMILES'])

print("Train set      : ", len(train_set))
print("Train label    : ", len(train_label))
print("Validate set   : ", len(valid_set))
print("Validate label : ", len(valid_label))
print("Real Train set : ", len(test_set))

# print(train_label)
train_label = [i.split(',') for i in train_label]
valid_label = [i.split(',') for i in valid_label]
# print(train_label)

##############
# Encode input
##############

SMILES_CHARS = [
    '#', '%', '(', ')', '+', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '=', '@',
    'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
    'R', 'S', 'T', 'V', 'X', 'Z',
    '[', '\\', ']',
    'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
    't', 'u']

smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))


def smiles_encoder(smiles, maxlen=240):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    X = np.zeros((maxlen, len(SMILES_CHARS)))
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
    return X


train_set_enc = np.array([smiles_encoder(i) for i in train_set])
valid_set_enc = np.array([smiles_encoder(i) for i in valid_set])
# np.savez_compressed("data/train_set_enc", mol=train_set_enc)
# np.savez_compressed("data/valid_set_enc", mol=valid_set_enc)


def smiles_decoder(X):
    smi = ''
    X = X.argmax(axis=-1)
    for i in X:
        smi += index2smi[i]
    return smi

# dec = smiles_decoder(mat[0])

################
# Encoder Label
################

def onehot_sentence(sentence):
    l = np.zeros(len(vocab))
    for label in sentence:
        l[vocab.index(label)] = 1
    return l

# trainint set
train_label_enc = np.zeros((len(train_label), len(vocab)), dtype=np.float32)

for i in range(len(train_label)):
    train_label_enc[i] = onehot_sentence(train_label[i])
print(train_label_enc.shape)

# valid set
valid_label_enc = np.zeros((len(valid_label), len(vocab)), dtype=np.float32)

for i in range(len(valid_label)):
    valid_label_enc[i] = onehot_sentence(valid_label[i])
print(valid_label_enc.shape)

# lab = LabelEncoder()
# lab.fit(vocab)
# print(lab.classes_)
# lab.inverse_transform()

# train_label_enc = [lab.transform(i) for i in train_label]
# valid_label_enc = [lab.transform(i) for i in valid_label]
# print(train_label_enc)

#################
# Build the model
#################

# data = np.load("data/train_set_enc.npz")
# mol = data['mol']
# print(mol.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(240, 55)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(109)
])

model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

print(len(train_set_enc))
print(len(train_label_enc))
print(train_label_enc)
model.fit(train_set_enc, train_label_enc, epochs=200)

####################
# Evaluate accuracy
####################

test_loss, test_acc = model.evaluate(valid_set_enc,  valid_label_enc, verbose=1)

print('\nTest accuracy:', test_acc)


##########
# Predict
##########
