import tensorflow as tf
from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import get_custom_objects
# get_custom_objects().update({'leaky_relu': tf.keras.layers.Activation(tf.nn.leaky_relu)})
# get_custom_objects().update({'leaky_relu': tf.keras.layers.Activation(tf.keras.layers.LeakyReLU)})

############
# Load model
############

model = load_model("model.h5", custom_objects=None, compile=True, options=None)

###############
# Load test set
###############

## Test data
test_set_fp = np.load("data/test_set_fingerprint_512bits_radius2.npz")['morgan']
test_set_vdw = np.load("data/test_set_vdW_volume.npz")["volume"]
test_set_enc = np.concatenate((test_set_fp, test_set_vdw), axis=1)

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
    labels_seq.append(",".join([labels[0], labels[1], labels[2]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[3]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[4]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[5]]))
    labels_seq.append(",".join([labels[0], labels[1], labels[6]]))
    # for i in range(0, 15, 3):
        # labels_seq.append(",".join(labels[i:(i+3)]))

    pred_for_sub.append(";".join(labels_seq))

# pprint(preds_clean)

test_set = pd.read_csv("data/test.csv")
test_set = list(test_set['SMILES'])
pred_label = {
    'SMILES': test_set,
    'PREDICTIONS': pred_for_sub
}
df = pd.DataFrame(pred_label)
# pprint(df)

submission_file_path = "submission.csv"

if SAVE_PREDICTION:
    print(f"Writing Submission (csv) to : {submission_file_path}")
    df.to_csv(
        submission_file_path,
        index=False
    )