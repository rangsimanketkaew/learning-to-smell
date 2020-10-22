import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Lipinski

train_set = pd.read_csv("data/train.csv")
train_set = list(train_set['SMILES'])

mol = Chem.MolFromSmiles(train_set[11])
print(rdMolDescriptors.CalcMolFormula(mol))

NumRotatableBonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
NumAliphaticCarbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
NumAromaticRings = rdMolDescriptors.CalcNumAromaticRings(mol)
NumAromaticHeterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
print(NumAliphaticCarbocycles)
print(NumAromaticRings)
print(NumAromaticHeterocycles)
