import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
from tensorflow.keras import backend as K


def jaccard_tensorflow(y_true, y_pred):
    """Jaccard score of Tensor in tensorflow for graph mode.
    """
    intersection = tf.sets.intersection(y_true[None:], y_pred[None:])
    intersection = tf.sparse.to_dense(intersection)[0]
    union = tf.sets.union(y_true[None:], y_pred[None:])
    union = tf.sparse.to_dense(union)[0]
    return float(len(intersection) / len(union))


def jaccard_tensorflow_eager(y_true, y_pred):
    """Jaccard score with built-in function in tensorflow in eager mode.
    """
    set1 = set(y_true.numpy())
    set2 = set(y_pred.numpy())
    return float((len(set1.intersection(set2))) / (len(set1.union(set2))))


def jaccard_from_keras_cont(y_true, y_pred):
    """Jaccard score for keras.
    Taken directly from https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return (1 - jac)
