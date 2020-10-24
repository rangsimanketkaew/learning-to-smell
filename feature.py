import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Lipinski

train_set = pd.read_csv("data/train.csv")
train_set = list(train_set['SMILES'])

mol = Chem.MolFromSmiles(train_set[11])
print(rdMolDescriptors.CalcMolFormula(mol))

MolWeight = Descriptors.MolWt(mol)
NumHAcceptors = Lipinski.NumHAcceptors(mol)
NumHDonors = Lipinski.NumHDonors(mol)
NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
NumAliphaticCarbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
NumAromaticHeterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
NumSaturatedRings = rdMolDescriptors.CalcNumSaturatedRings(mol)

print(MolWeight)
print(NumHAcceptors)
print(NumHDonors)
print(NumRotatableBonds)
print(NumAliphaticCarbocycles)
print(NumAromaticRings)
print(NumAromaticHeterocycles)
print(NumSaturatedRings)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    n = adjacency.shape[0]
    adjacency = adjacency + np.eye(n)
    degree = sum(adjacency)
    d_half = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
    return np.array(adjacency)

print(create_adjacency(mol).shape)
