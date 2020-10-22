import numpy as np
import pandas as pd
import scipy.sparse as sp

# import deepchem as dc
# RDkit for fingerprinting and cheminformatics
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
from rdkit.Chem.EState import Fingerprinter
# Sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split


def preprocess_graph(data):
    # The function is to preprocessed the adjacency matrix,
    # returning the normalized adjacency matrix in the form of numpy array for feeding into the model
    adj_ = data + sp.eye(data.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return np.array(adj_normalized)


def smiles_get_features(a):
    # This function will return the smiles code into list of feature for each atoms
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    atom_features = features.get_atom_features()  # initial atom feature vectors
    if atom_features.shape[0] > 60:
        return pd.np.nan
    return atom_features


def smiles_get_adj(a):  # This function retrieve the adjacency matrix from the molecule
    m = rdkit.Chem.MolFromSmiles(a)
    featurizer = dc.feat.ConvMolFeaturizer()
    features = featurizer.featurize([m])[0]
    if isinstance(features, np.ndarray):
        return pd.np.nan
    adj_list = features.get_adjacency_list()  # adjacency list (neighbor list)
    # convert adjacency list into adjacency matrix "A"
    adj = np.zeros((len(adj_list), len(adj_list)))
    if len(adj_list) > 60:
        return pd.np.nan
    return adj_list


def sim_graph(smile):
    mol = rdkit.Chem.MolFromSmiles(smile)
    if mol is None:
        return pd.np.nan
    Chem.Kekulize(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    am = Chem.GetAdjacencyMatrix(mol, useBO=True)
    if len(atoms) > 60:
        return pd.np.nan
    for i, atom in enumerate(atoms):
        am[i, i] = atom
    return am


def get_max_dim(d):  # This funcion is used to find the maximum dimension the set of data contain
    maxdim = 0
    for i in d:
        if i.shape[0] > maxdim:
            maxdim = i.shape[0]
    return maxdim


# This function is used to pad the data up to a given dimension
def pad_up_to(t, max_in_dims, constant_values=0):
    s = t.shape
    size = np.subtract(max_in_dims, s)
    return np.pad(t, ((0, size[0]), (0, size[1])), 'constant', constant_values=constant_values)

