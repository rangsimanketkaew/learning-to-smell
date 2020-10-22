####################
# Learning to smell
# Rangsiman Ketkaew
# October 2020
####################

import numpy as np
import pandas as pd
from pprint import pprint

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
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import ModelCheckpoint
# from keras.regularizers import WeightRegularizer
from keras.optimizers import Adam, SGD

# Plot
from matplotlib import pyplot as plt

###############
# Read dataset
###############
train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")
vocab = open("data/vocabulary.txt", 'r').read().split("\n")
sample_sub = pd.read_csv("data/sample_submission.csv")


###########
# Parameter
###########

N_EPOCHS = 200
BATCH_SIZE = 100
VALID_SPLIT = 0.2
LOSS = "binary_crossentropy"
METRICS = ['accuracy']
NAME_CHECKPOINT = 'model_checkpoint.hdf5'

###############
# Split dataset
###############

train_set, train_label = list(train_set['SMILES']), list(train_set['SENTENCE'])

# train_set, valid_set, train_label, valid_label = train_test_split(train_set, train_label, test_size=0.2, random_state=42)

test_set = list(test_set['SMILES'])

print("Train set      : ", len(train_set))
print("Train label    : ", len(train_label))
print("Real Train set : ", len(test_set))

# Get a flattened list of all labels
train_label = [i.split(',') for i in train_label]
train_label_sub = [item for sublist in train_label for item in sublist]

# count the number of occurences for each label
counts = dict((x, train_label_sub.count(x)) for x in set(train_label_sub))
pprint(counts)

##############
# Encode input
##############


# SMILES_CHARS = [
#     '#', '%', '(', ')', '+', '-', '.', '/',
#     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#     '=', '@',
#     'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
#     'R', 'S', 'T', 'V', 'X', 'Z',
#     '[', '\\', ']',
#     'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
#     't', 'u']

# smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
# index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))


# def smiles_encoder(smiles, maxlen=240):
#     smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
#     X = np.zeros((maxlen, len(SMILES_CHARS)))
#     for i, c in enumerate(smiles):
#         X[i, smi2index[c]] = 1
#     return X


# train_set_enc = np.array([smiles_encoder(i) for i in train_set])
# np.savez_compressed("data/train_set_enc", mol=train_set_enc)


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=8192)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    return npfp

def maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    return npfp


train_set_enc = np.array([morgan_fp(i) for i in train_set])
test_set_enc = np.array([morgan_fp(i) for i in test_set])

###############


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

# lab = LabelEncoder()
# lab.fit(vocab)
# print(lab.classes_)
# lab.inverse_transform()

# train_label_enc = [lab.transform(i) for i in train_label]
# print(train_label_enc)

#################
# Build the model
#################

# data = np.load("data/train_set_enc.npz")
# mol = data['mol']
# print(mol.shape)

model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(240, 55)),
    tf.keras.layers.Flatten(input_shape=(8192,)),
    # tf.keras.layers.Dropout(0.2, input_shape=(8192,)),
    tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dense(109, activation='sigmoid')
    # tf.keras.layers.Dense(109, activation='tanh')
])

opt_sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=opt_adam,
              loss=LOSS,
              metrics=METRICS,
              )
print(model.summary())

checkpointer = ModelCheckpoint(filepath=NAME_CHECKPOINT,
                               monitor='val_acc',
                               mode=max,
                               verbose=1,
                               save_best_only=False)

history = model.fit(
    train_set_enc,
    train_label_enc,
    validation_split=VALID_SPLIT,
    shuffle=False,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    use_multiprocessing=True,
    callbacks=[checkpointer]
)

# list all data in history
pprint(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  # show two plots

####################
# Evaluate accuracy
####################

# test_loss, test_acc = model.evaluate(valid_set_enc,  valid_label_enc, verbose=1)
# print('\nTest accuracy:', test_acc)


##########
# Predict
##########

preds = model.predict(test_set_enc)

# Choose the top 15 predictions for each sample and group by 3

ind2word = {i: x for i, x in enumerate(vocab)}

preds_clean = []
for i in range(preds.shape[0]):
    labels = [ind2word[i] for i in list(preds[i, :].argsort()[-15:][::-1])]

    labels_seq = []
    for i in range(0, 15, 3):
        labels_seq.append(",".join(labels[i:(i+3)]))

    preds_clean.append(";".join(labels_seq))

pprint(preds_clean)

# DataStructs.TanimotoSimilarity