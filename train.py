## Learning to smell

import numpy as np
import pandas as pd
import scipy.sparse as sp

# import deepchem as dc
import rdkit
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

def preprocess_graph(data): 
    #The function is to preprocessed the adjacency matrix, returning the normalized adjacency matrix in the form of numpy array for feeding into the model
    adj_ = data + sp.eye(data.shape[0]) 
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return np.array(adj_normalized)

def smiles_get_features(a): 
    #This function will return the smiles code into list of feature for each atoms
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    atom_features = features.get_atom_features() # initial atom feature vectors
    if atom_features.shape[0] > 60:
        return pd.np.nan
    return atom_features

def smiles_get_adj(a): #This function retrieve the adjacency matrix from the molecule
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    adj_list = features.get_adjacency_list() # adjacency list (neighbor list)
    adj=np.zeros((len(adj_list), len(adj_list))) # convert adjacency list into adjacency matrix "A"
    if len(adj_list) > 60:
        return pd.np.nan
    return adj_list

def sim_graph(smile):
    mol = rdkit.Chem.MolFromSmiles(smile)
    if mol is None:
        return pd.np.nan
    Chem.Kekulize(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    am = Chem.GetAdjacencyMatrix(mol,useBO=True)
    if len(atoms)>60:
        return pd.np.nan
    for i,atom in enumerate(atoms):
        am[i,i] = atom
    return am

def get_max_dim(d): #This funcion is used to find the maximum dimension the set of data contain
    maxdim = 0
    for i in d:
        if i.shape[0]>maxdim:
            maxdim = i.shape[0]
    return maxdim

def pad_up_to(t, max_in_dims, constant_values=0): #This function is used to pad the data up to a given dimension
    s = t.shape
    size = np.subtract(max_in_dims, s)
    return np.pad(t, ((0,size[0]),(0,size[1])), 'constant', constant_values=constant_values)

# read dataset

train_set = pd.read_csv("data/train.csv")
test_set = pd.read_csv("data/test.csv")
sample_sub = pd.read_csv("data/sample_submission.csv")
vocab = open("data/vocabulary.txt", 'r').read().split("\n")

print(f"size of training {train_set.shape}")
print(f"size of testing {test_set.shape}")

def onehot_sentence(sentence):
    l = np.zeros(len(vocab))
    for label in sentence.split(','):
        l[vocab.index(label)] = 1
    return l

print(train_set.SENTENCE)

test_vocab = np.zeros((train_set.shape[0], len(vocab)), dtype=np.float32)

for i in range(train_set.shape[0]):
    test_vocab[i] = onehot_sentence(train_set.SENTENCE.iloc[i])

train_set, train_label = list(train_set['SMILES']), list(train_set['SENTENCE'])

for i in train_set: print(i)

train_label = [i.split(',') for i in train_label]
print(train_label[-1])
# print(vocab)