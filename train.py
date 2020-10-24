####################
# Learning to smell
# Rangsiman Ketkaew
# October 2020
####################

import numpy as np
import pandas as pd
from pprint import pprint

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
# Plot
from matplotlib import pyplot as plt


#################
# Hyper parameter
#################

N_EPOCHS = 500
BATCH_SIZE = 100
ACT_HIDDEN = LeakyReLU(alpha=0.1)
ACT_OUTPUT = 'sigmoid'
DROPOUT = 0.2
KERNEL_REG = l1_l2(l1=1e-5, l2=1e-4)
BIAS_REG = l2(1e-4)
ACTI_REG = l2(1e-5)
VALID_SPLIT = 0.2

# OPTIMIZER = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# tf.config.run_functions_eagerly(True)
def my_hamming(y_true, y_pred):
    """Hamming Loss"""
    return tfa.metrics.hamming.hamming_loss_fn(y_true=y_true, y_pred=y_pred, mode="multiclass", threshold=0.8)

def my_npair(y_true, y_pred):
    """NPair Loss"""
    return tfa.losses.npairs_multilabel_loss(y_true=y_true, y_pred=y_pred)

LOSS = "binary_crossentropy"
# LOSS = tf.nn.sigmoid_cross_entropy_with_logits

METRICS = ['accuracy']
NAME_CHECKPOINT = 'model_checkpoint.hdf5'
SAVE_SUBMISSION = True

##########
# Callback
##########

checkpointer = ModelCheckpoint(
    filepath=NAME_CHECKPOINT,
    monitor='val_acc',
    mode='max',
    verbose=0,
    save_best_only=True,
    save_weights_only=False
    )
reduce_lr = ReduceLROnPlateau(
    monitor='loss', 
    factor=0.1, 
    patience=5, 
    min_lr=0.00001, 
    verbose=1
    )
earlystop = EarlyStopping(
    monitor='val_loss', 
    min_delta=0,
    patience=50, 
    mode='auto', 
    verbose=1
    )

cb = [
    checkpointer, 
    # reduce_lr, 
    # earlystop
    ]

######
# plot
######

HIST_ACC = 'accuracy'
HIST_VAL_ACC = 'val_accuracy'
HIST_LOSS = 'loss'
HIST_VAL_LOSS = 'val_loss'

############################################################################

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
# Extract dataset
#################

train_set, train_label = list(train_set['SMILES']), list(train_set['SENTENCE'])
test_set = list(test_set['SMILES'])

# Get a flattened list of all labels
train_label = [i.split(',') for i in train_label]

# count the number of occurences for each label
# train_label_sub = [item for sublist in train_label for item in sublist]
# counts = dict((x, train_label_sub.count(x)) for x in set(train_label_sub))
# pprint(counts)

# Split dataset
train_set, valid_set, train_label, valid_label = train_test_split(train_set, train_label, test_size=0.2, random_state=42)

print("Train set      : ", len(train_set))
print("Train label    : ", len(train_label))
print("Valid set      : ", len(valid_set))
print("Valid label    : ", len(valid_label))
print("Real Train set : ", len(test_set))


##############
# Encode input
##############

#===================
# 1) Simple encoding
#===================

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

# def smiles_decoder(X):
#     smi = ''
#     X = X.argmax(axis=-1)
#     for i in X:
#         smi += index2smi[i]
#     return smi

# dec = smiles_decoder(mat[0])

#===================
# 2) Morgan encoding
#===================


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
valid_set_enc = np.array([morgan_fp(i) for i in valid_set])
test_set_enc = np.array([morgan_fp(i) for i in test_set])


################
# Encoder Label
################


def onehot_sentence(sentence):
    l = np.zeros(len(vocab))
    for label in sentence:
        l[vocab.index(label)] = 1
    return l


train_label_enc = np.zeros((len(train_label), len(vocab)), dtype=np.float32)
for i in range(len(train_label)):
    train_label_enc[i] = onehot_sentence(train_label[i])

valid_label_enc = np.zeros((len(valid_label), len(vocab)), dtype=np.float32)
for i in range(len(valid_label)):
    valid_label_enc[i] = onehot_sentence(valid_label[i])

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

model = Sequential([
    Flatten(input_shape=(8192,)),
    # Dropout(0.2, input_shape=(8192,)),
    Dense(128, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(128, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT/2),
    BatchNormalization(),
    Dense(128, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT/4),
    BatchNormalization(),
    Dense(109, activation=ACT_OUTPUT),
])

model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS,
              )


history = model.fit(
    train_set_enc,
    train_label_enc,
    # validation_split=VALID_SPLIT,
    validation_data=(valid_set_enc, valid_label_enc),
    shuffle=False,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    use_multiprocessing=True,
    verbose=1,
    callbacks=[cb]
)


print(model.summary())
# list all data in history
# pprint(history.history.keys())
print("")
print(f"Last Accuracy     : {history.history[HIST_ACC][-1]}")
print(f"Max  Accuracy     : {np.max(history.history[HIST_ACC])}")
print(f"Last Val accuracy : {history.history[HIST_VAL_ACC][-1]}")
print(f"Max  Val accuracy : {np.max(history.history[HIST_VAL_ACC])}")
print("-----------")
print(f"Min loss         : {np.min(history.history[HIST_LOSS])}")
print(f"Min val loss     : {np.min(history.history[HIST_VAL_LOSS])}")

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history[HIST_ACC])
plt.plot(history.history[HIST_VAL_ACC])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.figure(2)
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

if SAVE_SUBMISSION:
    print("Writing Submission (csv) to : ", submission_file_path)
    df.to_csv(
        submission_file_path,
        index=False
    )

# DataStructs.TanimotoSimilarity
