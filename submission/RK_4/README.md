## Model details

- Evaluation result: https://www.aicrowd.com/challenges/learning-to-smell/submissions/92222
- Number of training sample: 249645
- 6 LeakyReLu hidden layers each with 2048 nodes
- Output layer uses sigmoid
- No batch normalization

```
#################
# Hyper parameter
#################

N_EPOCHS = 50
BATCH_SIZE = 100
VALID_SPLIT = 0.2
DROPOUT = 0.2
KERNEL_REG = l1_l2(l1=1e-5, l2=1e-4)
BIAS_REG = l2(1e-4)
ACTI_REG = l2(1e-5)
OPTIMIZER = opt_adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
LOSS = "binary_crossentropy"
METRICS = ['accuracy']
NAME_CHECKPOINT = 'model_checkpoint.hdf5'
```