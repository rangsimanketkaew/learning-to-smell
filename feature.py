import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Lipinski
from rdkit.Chem import rdchem

from vdW_volume import get_vdw

train_set = pd.read_csv("data/train.csv")
smiles_set = list(train_set['SMILES'])
sentence_set = list(train_set['SENTENCE'])
# print(sentence_set[0])

###########################################
### Randomly generate SMILES - augmentation
###########################################

all_smiles = []
with open(f"data/augmentation/train_aug_random100.csv", "w") as f:
    f.write("SMILES,SENTENCE\n")
    for i, smile in enumerate(smiles_set):
        mol = Chem.MolFromSmiles(smile)
        smiles = []
        ## Insert original smiles
        smiles.append(smile)
        for _ in range(100):
            # smi = Chem.MolToSmiles(mol, doRandom=True)
            try:
                smi = Chem.MolToSmiles(mol, doRandom=True)
            except:
                pass
            # print(smi)
            smiles.append(smi)
        print("Mol ", i, "done")
        ## Remove duplicate smiles
        smiles = list(set(smiles))
        ## Write smiles and its sentence to file
        for j in smiles:
            f.write(f'{j},"{sentence_set[i]}"\n')

    f.close()

# print(rdMolDescriptors.CalcMolFormula(mol))

# MolWeight = Descriptors.MolWt(mol)
# NumHAcceptors = Lipinski.NumHAcceptors(mol)
# NumHDonors = Lipinski.NumHDonors(mol)
# NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
# NumAliphaticCarbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
# NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
# NumAromaticHeterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
# NumSaturatedRings = rdMolDescriptors.CalcNumSaturatedRings(mol)

# print(MolWeight)
# print(NumHAcceptors)
# print(NumHDonors)
# print(NumRotatableBonds)
# print(NumAliphaticCarbocycles)
# print(NumAromaticRings)
# print(NumAromaticHeterocycles)
# print(NumSaturatedRings)
# print("---")
# print(mol.GetNumAtoms())

#######################
### Generate vdW volume
#######################
# with open("vdW_volume.txt", 'w') as f:
#     for i in train_set:
#         f.write(f"{get_vdw(i)}\n")
#         print(get_vdw(i))
# f.close

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
