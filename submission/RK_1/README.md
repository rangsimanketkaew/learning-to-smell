## Model details

- Evaluation result: https://www.aicrowd.com/challenges/learning-to-smell/submissions/91214
- 3 LeakyReLu hidden layers each with 256 nodes
- Output layer uses sigmoid
- No batch normalization

```
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
```