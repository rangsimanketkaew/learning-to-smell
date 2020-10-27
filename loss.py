import tensorflow_addons as tfa
from tensorflow.keras import backend as K


def humming_loss(y_true, y_pred):
    # tf.config.run_functions_eagerly(True)
    """Hamming Loss"""
    return tfa.metrics.hamming.hamming_loss_fn(y_true=y_true, y_pred=y_pred, mode="multiclass", threshold=0.8)


def npair_loss(y_true, y_pred):
    """NPair Loss"""
    return tfa.losses.npairs_multilabel_loss(y_true=y_true, y_pred=y_pred)


def jaccard_loss(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    Reference:  
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
