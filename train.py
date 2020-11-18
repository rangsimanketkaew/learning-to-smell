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
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import multi_gpu_model
# Plot
from matplotlib import pyplot as plt

import loss, metric
from func import onehot_sentence

# Fix GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


################################# SETTINGS #################################
# Hyper parameter
N_EPOCHS = 200
BATCH_SIZE = 32
ACT_HIDDEN = LeakyReLU(alpha=0.2)
ACT_OUTPUT = 'sigmoid'
DROPOUT = 0.2
KERNEL_REG = l1_l2(l1=1e-5, l2=1e-4)
BIAS_REG = l2(1e-4)
ACTI_REG = l2(1e-5)
TRAIN_WITH_VALID = True
VALID_SPLIT = 0.9
# GPU = 2

# OPTIMIZER = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# OPTIMIZER = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# OPTIMIZER = Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07, name="Adagrad")

LOSS = "binary_crossentropy"
# LOSS = "categorical_crossentropy"
# LOSS = "KLDivergence"
# LOSS = loss.jaccard_loss
# LOSS = tf.nn.sigmoid_cross_entropy_with_logits

METRICS = ['accuracy']
# METRICS = [metric.jaccard_5sentences]
NAME_CHECKPOINT = 'model_checkpoint.h5'
PATH_SAVE_MODEL = 'model.h5'
SAVE_PREDICTION = True
SHOW_FIGURE = True

if os.name == "posix": os.system("export HDF5_USE_FILE_LOCKING=FALSE")

# Callback
checkpointer = ModelCheckpoint(filepath=NAME_CHECKPOINT, monitor='val_acc', mode='max',
                               verbose=0, save_best_only=False, save_weights_only=False)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=30, mode='auto', verbose=1)
# monitor='val_loss' can also be used if train with validation

cb = [checkpointer, reduce_lr, earlystop]

# plot
HIST_ACC, HIST_VAL_ACC = 'accuracy', 'val_accuracy'
HIST_LOSS, HIST_VAL_LOSS = 'loss', 'val_loss'

submission_file_path = "submission.csv"

#---------------------------------------

## Training data
train_set_enc = np.load("data/train_set_all_descriptors.npz")["features"]
## Test data
test_set_enc = np.load("data/test_set_all_descriptors.npz")["features"]

############################################################################

## Train label
train_label = list(pd.read_csv("data/train.csv")['SENTENCE'])
# train_label = list(pd.read_csv("data/augmentation/train_aug_random20.csv")['SENTENCE'])
train_label = [i.split(',') for i in train_label]
vocab = open("data/vocabulary.txt", 'r').read().split("\n")
train_label_enc = np.zeros((len(train_label), len(vocab)), dtype=np.float32)
for i in range(len(train_label)):
    train_label_enc[i] = onehot_sentence(vocab, train_label[i])

#---------------------------------------

print(train_set_enc.shape)
print(test_set_enc.shape)
print(train_label_enc.shape)
assert train_set_enc.shape[0] == train_label_enc.shape[0]
train_set, train_label = train_set_enc, train_label_enc

## Shuffle dataset before splitting
# index = np.arange(train_set_enc.shape[0])
# np.random.shuffle(index)
# train_set_enc, train_label_enc = train_set_enc[index], train_label_enc[index]

# # Split train set --> real train set + validation set
# train_set, valid_set, train_label, valid_label = train_test_split(train_set_enc, train_label_enc, test_size=1-VALID_SPLIT, random_state=0)
# validation_train_test = (valid_set, valid_label)

#################
# Build the model
#################

model = Sequential([
    Flatten(input_shape=(test_set_enc.shape[1],)),
    # Dropout(0.2, input_shape=(8192,)),
    Dense(128, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(128, activation=ACT_HIDDEN, kernel_regularizer=KERNEL_REG, bias_regularizer=BIAS_REG, activity_regularizer=ACTI_REG),
    Dense(109, activation=ACT_OUTPUT),
])

model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
model.summary()

# Now can train only on a single GPU - failed with multi-GPUs. Needs to be fixed!
history = model.fit(
    train_set,
    train_label,
    validation_split=VALID_SPLIT,
    # validation_data=(valid_set, valid_label),
    shuffle=False,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    use_multiprocessing=True,
    verbose=1,
    callbacks=[cb]
)

# print(model.summary())

# Save model
save_model(model, PATH_SAVE_MODEL, overwrite=True, include_optimizer=True, save_format='h5', signatures=None, options=None)

print(f"Model has been saved to {PATH_SAVE_MODEL}")
print(f"Checkpoint has been saved to {NAME_CHECKPOINT}")

# list all data in history
# pprint(history.history.keys())
print(f"\nLast Accuracy     : {history.history[HIST_ACC][-1]}")
print(f"Max  Accuracy     : {np.max(history.history[HIST_ACC])}")
if TRAIN_WITH_VALID: print(f"Last Val accuracy : {history.history[HIST_VAL_ACC][-1]}")
if TRAIN_WITH_VALID: print(f"Max  Val accuracy : {np.max(history.history[HIST_VAL_ACC])}")
print("-----------")
print(f"Min loss         : {np.min(history.history[HIST_LOSS])}")
if TRAIN_WITH_VALID: print(f"Min val loss     : {np.min(history.history[HIST_VAL_LOSS])}")

if SHOW_FIGURE:
    # summarize history for accuracy
    plt.figure(1)
    plt.plot(history.history[HIST_ACC])
    if TRAIN_WITH_VALID: plt.plot(history.history[HIST_VAL_ACC])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history[HIST_LOSS])
    if TRAIN_WITH_VALID: plt.plot(history.history[HIST_VAL_LOSS])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
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
    # for i in range(0, 15, 3):
    #     labels_seq.append(",".join(labels[i:(i+3)]))
    labels_seq.append(",".join([labels[0], labels[1], labels[2]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[3]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[4]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[5]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[6]]))
    pred_for_sub.append(";".join(labels_seq))

test_set = pd.read_csv("data/test.csv")
test_set = list(test_set['SMILES'])
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
