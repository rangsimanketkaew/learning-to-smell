import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras import backend as K

def jaccard_5sentences(y_true, y_pred):
    """Jaccard score for judging the performance of model
    Compare all 15 labels with the ground-truths.
    """
    set1 = set(y_true.numpy())
    set2 = set(y_pred.numpy())
    jac = float((len(set1.intersection(set2))) / (len(set1.union(set2))))
    return jac

def jaccard_5sentences_tensor(y_true, y_pred):
    """Jaccard score for judging the performance of model
    Compare all 15 labels with the ground-truths.
    """
    print(type(y_true))
    print(type(y_pred))
    intersection = tf.sets.intersection(y_true[None:], y_pred[None:])
    intersection = tf.sparse.to_dense(intersection)[0]
    union = tf.sets.union(y_true[None:], y_pred[None:])
    union = tf.sparse.to_dense(union)[0]
    jac = float(len(intersection) / len(union))
    return jac

def jaccard_each_sentences(y_true, y_pred):
    """Jaccard score for judging the performance of model
    Compare each sentence (3 labels) with the ground-truths and choose the highest score.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection) / (sum_ - intersection)
    return (1 - jac)