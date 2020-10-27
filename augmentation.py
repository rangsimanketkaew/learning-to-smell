import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem

###########################################
### Randomly generate SMILES - augmentation
###########################################

# training set
train_set = pd.read_csv("data/train.csv")
smiles_set = list(train_set['SMILES'])
sentence_set = list(train_set['SENTENCE'])

all_smiles = []
with open(f"data/augmentation/train_aug_random20.csv", "w") as f:
    f.write("SMILES,SENTENCE\n")
    for i, smile in enumerate(smiles_set):
        mol = Chem.MolFromSmiles(smile)
        smiles = []
        ## Insert original smiles
        smiles.append(smile)
        for _ in range(20):
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