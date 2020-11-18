import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Lipinski
from rdkit.Chem import rdchem, Fragments

from vdW_volume import get_vdw

# training set
train_set = pd.read_csv("data/train.csv")
smiles_set = list(train_set['SMILES'])
sentence_set = list(train_set['SENTENCE'])

# test set
test_set = pd.read_csv("data/test.csv")
smiles_test_set = test_set['SMILES']

## ============================================== ##

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    n = adjacency.shape[0]
    adjacency = adjacency + np.eye(n)
    degree = sum(adjacency)
    d_half = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
    return np.array(adjacency)

# print(create_adjacency(mol))

### Generate vdW volume
# with open("data/test_set_vdW_volume.txt", 'w') as f:
#     for i in smiles_test_set:
#         f.write(f"{get_vdw(i)}\n")
#         print(get_vdw(i))
# f.close

#----------------------------------------------

train_feat = []
frag = []
for i in smiles_set:
    mol = Chem.MolFromSmiles(i)
    MolWeight = Descriptors.MolWt(mol)
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    NumHAcceptors = Lipinski.NumHAcceptors(mol)
    NumHDonors = Lipinski.NumHDonors(mol)
    NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    NumAliphaticCarbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
    NumAromaticHeterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    NumSaturatedRings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    a = [MolWeight, HeavyAtomMolWt, MaxAbsPartialCharge, MinAbsPartialCharge, NumRadicalElectrons, 
        NumValenceElectrons, NumHAcceptors, NumHDonors, NumRotatableBonds, 
        NumAliphaticCarbocycles, NumAromaticRings, NumAromaticHeterocycles, NumSaturatedRings]
    train_feat.append(a)

    b = []
    for name, val in Fragments.__dict__.items():
        if name[:3] == 'fr_' and callable(val):
            # print(val(mol))
            b.append(val(mol))
    frag.append(b)

train_feat = np.array(train_feat)
train_frag = np.array(frag)
print(train_feat.shape)
print(train_frag.shape)
vol = np.loadtxt("data/train_set_vdW_volume.txt").reshape(-1, 1)
print(vol.shape)
# np.savez_compressed("data/train_set_vdW_volume.npz", volume=vol)
train_feat = np.concatenate((train_feat, train_frag, vol), axis=1)
print(train_feat.shape)
np.savez_compressed("data/train_set_all_descriptors.npz", features=train_feat)

#----------------------------------------------

test_feat = []
frag = []
for i in smiles_test_set:
    mol = Chem.MolFromSmiles(i)
    MolWeight = Descriptors.MolWt(mol)
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    NumHAcceptors = Lipinski.NumHAcceptors(mol)
    NumHDonors = Lipinski.NumHDonors(mol)
    NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    NumAliphaticCarbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
    NumAromaticHeterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    NumSaturatedRings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    a = [MolWeight, HeavyAtomMolWt, MaxAbsPartialCharge, MinAbsPartialCharge, NumRadicalElectrons, 
        NumValenceElectrons, NumHAcceptors, NumHDonors, NumRotatableBonds, 
        NumAliphaticCarbocycles, NumAromaticRings, NumAromaticHeterocycles, NumSaturatedRings]
    test_feat.append(a)

    b = []
    for name, val in Fragments.__dict__.items():
        if name[:3] == 'fr_' and callable(val):
            # print(val(mol))
            b.append(val(mol))
    frag.append(b)

test_feat = np.array(test_feat)
test_frag = np.array(frag)
print(test_feat.shape)
print(test_frag.shape)
vol = np.loadtxt("data/test_set_vdW_volume.txt").reshape(-1, 1)
print(vol.shape)
# np.savez_compressed("data/test_set_vdW_volume.npz", volume=vol)
test_feat = np.concatenate((test_feat, test_frag, vol), axis=1)
print(test_feat.shape)
np.savez_compressed("data/test_set_all_descriptors.npz", features=test_feat)

#----------------------------------------------
