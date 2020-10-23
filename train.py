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
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from keras.regularizers import WeightRegularizer
from keras.optimizers import Adam, SGD
from keras.regularizers import l1, l2, l1_l2

# Plot
from matplotlib import pyplot as plt

###############
# Read dataset
###############
train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")
vocab = open("data/vocabulary.txt", 'r').read().split("\n")
sample_sub = pd.read_csv("data/sample_submission.csv")

# submission file name
submission_file_path = "submission.csv"

#################
# Hyper parameter
#################

N_EPOCHS = 200
BATCH_SIZE = 100
VALID_SPLIT = 0.2
DROPOUT = 0.2
KERNEL_REG = l1_l2(l1=1e-5, l2=1e-4)
BIAS_REG = l2(1e-4)
ACTI_REG = l2(1e-5)
# OPTIMIZR = opt_sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
OPTIMIZER = opt_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# tf.config.run_functions_eagerly(True)
def my_hamming(y_true, y_pred):
    return tfa.metrics.hamming.hamming_loss_fn(y_true=y_true, y_pred=y_pred, mode="multiclass", threshold=0.8)
# LOSS = my_hamming

def my_npair(y_true, y_pred):
    return tfa.losses.npairs_multilabel_loss(y_true=y_true, y_pred=y_pred)

LOSS = "binary_crossentropy"
# LOSS = tf.nn.sigmoid_cross_entropy_with_logits
# METRICS = ['accuracy']
METRICS = ['accuracy']
NAME_CHECKPOINT = 'model_checkpoint.hdf5'

# plot
HIST_ACC = 'accuracy'
HIST_VAL_ACC = 'val_accuracy'
HIST_LOSS = 'loss'
HIST_VAL_LOSS = 'val_loss'

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
    tf.keras.layers.Dense(
        256, 
        activation=LeakyReLU(alpha=0.1), 
        kernel_regularizer=KERNEL_REG, 
        bias_regularizer=BIAS_REG, 
        activity_regularizer=ACTI_REG
        ),
    tf.keras.layers.Dropout(DROPOUT),
    # tf.keras.layers.BatchNormalization(),
    #---
    tf.keras.layers.Dense(
        256, 
        activation=LeakyReLU(alpha=0.1), 
        kernel_regularizer=KERNEL_REG, 
        bias_regularizer=BIAS_REG, 
        activity_regularizer=ACTI_REG
        ),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.BatchNormalization(),
    #---
    tf.keras.layers.Dense(
        256, 
        activation=LeakyReLU(alpha=0.1), 
        kernel_regularizer=KERNEL_REG, 
        bias_regularizer=BIAS_REG, 
        activity_regularizer=ACTI_REG
        ),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.BatchNormalization(),
    # --- Output layer
    tf.keras.layers.Dense(109, activation='sigmoid'),
    # tf.keras.layers.Dense(109, activation='softmax')
    # tf.keras.layers.BatchNormalization()
])

model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS,
              )

print(model.summary())

# Callback
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=50, min_lr=0.00001, verbose=1)
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
# pprint(history.history.keys())

print("")
print(f"Max accuracy     : {np.max(history.history[HIST_ACC])}")
print(f"Max val accuracy : {np.max(history.history[HIST_VAL_ACC])}")
print(f"Min loss         : {np.min(history.history[HIST_LOSS])}")
print(f"Min val loss     : {np.min(history.history[HIST_VAL_LOSS])}")

# summarize history for accuracy
plt.plot(history.history[HIST_ACC])
plt.plot(history.history[HIST_VAL_ACC])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history[HIST_LOSS])
plt.plot(history.history[HIST_VAL_LOSS])
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

pred = model.predict(test_set_enc)

# Choose the top 15 predictions for each sample and group by 3

ind2word = {i: x for i, x in enumerate(vocab)}

pred_for_sub = []
for i in range(pred.shape[0]):
    labels = [ind2word[i] for i in list(pred[i, :].argsort()[-15:][::-1])]

    labels_seq = []
    for i in range(0, 15, 3):
        labels_seq.append(",".join(labels[i:(i+3)]))

    pred_for_sub.append(";".join(labels_seq))


# pprint(preds_clean)

pred_label = {
    'SMILES': test_set,
    'PREDICTIONS': pred_for_sub
}
df = pd.DataFrame(pred_label)
# pprint(df)

print("Writing Sample Submission to : ", submission_file_path)
df.to_csv(
    submission_file_path,
    index=False
)

# DataStructs.TanimotoSimilarity