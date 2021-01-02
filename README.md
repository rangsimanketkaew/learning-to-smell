# Learning to Smell Challenge

This repository is dedicated to [Learning to Smell challenge](https://www.aicrowd.com/challenges/learning-to-smell) organized by AIcrowd. My best Jaccard score prediction is 0.346 (the 1st round).

## Challenge

Use machine learning (or any other related methods/techniques) to build a model and train it with the available dataset (4316 molecules) and predict smell character of molecules in a given test set (1079 molecules). There are 109 odors present in the dataset ([data/vocabulary.txt](data/vocabulary.txt)). More details can be found on the website of the competition.

#### Dataset
1. train.csv - (4316 molecules) : This csv file contains the attributes describing the molecules along with their "Sentence"
2. test.csv - (1079 molecules) (Round-1) : File that will be used for actual evaluation for the leaderboard score but does not have the "Sentence" for molecules.
3. vocabulary.txt : A file containing the list of all odors present in the dataset

#### Evaluation

Jaccard (Tanimoto) score - [Read more](https://en.wikipedia.org/wiki/Jaccard_index)

## My neural network

My neural network is designed with a feedforward neural network using Keras/TensorFlow library. The features (descriptors) that I used for training models are the Morgan fingerprint and structural properties of a molecule such as functional groups, number of benzene rings, van der Waals volume, etc.

## Source code structures

In alphabetical order

- [AdamW.py](AdamW.py) - Implementation of AdamW optimizer
- [augmentation.py](augmentation.py) - Data augmentation
- [descriptors.py](descriptors.py) - Extract structural properties from molecules and use them as features
- [fingerprint.py](fingerprint.py) - Calculate Morgan fingerprint
- [func.py](func.py) - Miscellaneous functions
- [loss.py](loss.py) - Customized loss functions
- [metric.py](metric.py) - Customized metric
- [predict.py](predict.py) - Prediction
- [README.md](README.md) - This file
- [run_24c_1g.sh](run_24c_1g.sh) SLURM input file
- [train-predict.py](train-predict.py) - Google Colab file
- [train.py](train.py) - Train model
- [vdW_volume.py](vdW_volume.py) - Calculate van der Waals volume
- [data](data) - Dataset and files
- [submission](submission) - Submission files

## Steps to reproduce my score

1. Calculate Morgan fingerprints with 8192 bits using RDKit.
2. Build a model containing 3 hidden layers with 128 neurons each.
3. DropOut and batch normalization are also applied.
4. Compile model with Adam optimizer. Use Categorical entropy as a loss function and accuracy as a metric.
5. Train model for 300 epochs. Learning rate schedule and early stop techniques are also applied when a metric has stopped improving.
6. Predict the smells.
7. Choose the top 15 smell predictions for each sample (molecule) and group by 3.
8. Submit the prediction results (.csv) and get the score. Example of a submission file is [this file](data/submission/RK_1/submission.csv)
