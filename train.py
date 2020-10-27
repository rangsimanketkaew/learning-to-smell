####################
# Learning to smell
# Rangsiman Ketkaew
# October 2020
####################

import os
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
# tf.compat.v1.enable_eager_execution()  # usually turn on by default
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import multi_gpu_model
# Plot
from matplotlib import pyplot as plt

import loss, metric
from func import onehot_sentence


# Hyper parameter
N_EPOCHS = 100
BATCH_SIZE = 100
ACT_HIDDEN = LeakyReLU(alpha=0.1)
ACT_OUTPUT = 'sigmoid'
DROPOUT = 0.2
KERNEL_REG = l1_l2(l1=1e-5, l2=1e-4)
BIAS_REG = l2(1e-4)
ACTI_REG = l2(1e-5)
VALID_SPLIT = 0.001
# GPU = 2

# OPTIMIZER = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

LOSS = "binary_crossentropy"
# LOSS = loss.jaccard_loss
# LOSS = tf.nn.sigmoid_cross_entropy_with_logits

METRICS = ['accuracy']
# METRICS = [metric.jaccard_5sentences]
NAME_CHECKPOINT = 'model_checkpoint.hdf5'
PATH_SAVE_MODEL = 'model.hdf5'
SAVE_PREDICTION = True
SHOW_FIGURE = False

if os.name == "posix":
    os.system("export HDF5_USE_FILE_LOCKING=FALSE")

# Callback
checkpointer = ModelCheckpoint(filepath=NAME_CHECKPOINT, monitor='val_acc', mode='max',
                               verbose=0, save_best_only=False, save_weights_only=False)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, mode='auto', verbose=1)

cb = [checkpointer, reduce_lr, earlystop]

# plot
HIST_ACC, HIST_VAL_ACC = 'accuracy', 'val_accuracy'
HIST_LOSS, HIST_VAL_LOSS = 'loss', 'val_loss'

submission_file_path = "submission.csv"

#---------------------------------------

## Training data
train_set_enc = np.load("data/train_set_fingerprint_8192bits.npz")['morgan']

## Test data
test_set = pd.read_csv("data/test.csv")
test_set = list(test_set['SMILES'])
test_set_enc = np.load("data/test_set_fingerprint_8192bits.npz")['morgan']

## Train label
vocab = open("data/vocabulary.txt", 'r').read().split("\n")
train_label = list(pd.read_csv("data/train.csv")['SENTENCE'])
train_label = [i.split(',') for i in train_label]
train_label_enc = np.zeros((len(train_label), len(vocab)), dtype=np.float32)
for i in range(len(train_label)):
    train_label_enc[i] = onehot_sentence(vocab, train_label[i])

#---------------------------------------

## Shuffle dataset before splitting
assert train_set_enc.shape[0] == train_label_enc.shape[0]
index = np.arange(train_set_enc.shape[0])
np.random.shuffle(index)
train_set_enc, train_label_enc = train_set_enc[index], train_label_enc[index]

# Split train set --> real train set + validation set
train_set, valid_set, train_label, valid_label = train_test_split(train_set_enc, train_label_enc, test_size=VALID_SPLIT, random_state=0)

############################################################################

#################
# Build the model
#################

# data = np.load("data/train_set_enc.npz")
# mol = data['mol']
# print(mol.shape)

model = Sequential([
    Flatten(input_shape=(8192,)),
    # Dropout(0.2, input_shape=(8192,)),
    Dense(256, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(256, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(256, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT),
    BatchNormalization(),
    # Dense(1028, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    # Dropout(DROPOUT),
    # BatchNormalization(),
    # Dense(1028, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    # Dropout(DROPOUT),
    # BatchNormalization(),
    # Dense(1028, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    # Dropout(DROPOUT),
    # BatchNormalization(),
    # Dense(1028, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    # Dropout(DROPOUT),
    # BatchNormalization(),
    # Dense(1028, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    # Dropout(DROPOUT),
    # BatchNormalization(),
    Dense(109, activation=ACT_OUTPUT),
])

# Now can train only on a single GPU - failed with multi-GPUs. Needs to be fixed!
#if GPU > 1:
#    # disable eager execution
#    tf.compat.v1.disable_eager_execution()
#    print("[INFO] training with {} GPUs...".format(GPU))
#    # we'll store a copy of the model on *every* GPU and then combine
#    # the results from the gradient updates on the CPU
#    # tf.device("/cpu:0")
#    # tf.compat.v1.reset_default_graph() 
#    # backend.clear_session()
#    # make the model parallel
#    model = multi_gpu_model(model, gpus=GPU)

model.compile(optimizer=OPTIMIZER,
              loss=LOSS,
              metrics=METRICS,
            #   run_eagerly=True
              )


history = model.fit(
    train_set,
    train_label,
    # validation_split=VALID_SPLIT,
    validation_data=(valid_set, valid_label),
    shuffle=False,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    use_multiprocessing=True,
    verbose=1,
    callbacks=[cb]
)

print(model.summary())

# Save model
save_model(
    model, PATH_SAVE_MODEL, overwrite=True, include_optimizer=True, save_format='h5',
    signatures=None, options=None
)

print(f"Model has been saved to {PATH_SAVE_MODEL}")
print(f"Checkpoint has been saved to {NAME_CHECKPOINT}")

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

if SHOW_FIGURE:
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

if SAVE_PREDICTION:
    print(f"Writing Submission (csv) to : {submission_file_path}")
    df.to_csv(
        submission_file_path,
        index=False
    )

# DataStructs.TanimotoSimilarity
