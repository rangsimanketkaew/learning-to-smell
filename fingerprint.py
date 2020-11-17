import numpy as np
import pandas as pd

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def smiles_encoder(smiles, maxlen=240):
    """Calculate simple encoder from SMILES

    Example:
    >>> train_set_enc = np.array([smiles_encoder(i) for i in train_set])
    """
    SMILES_CHARS = [
    '#', '%', '(', ')', '+', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '=', '@',
    'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
    'R', 'S', 'T', 'V', 'X', 'Z',
    '[', '\\', ']',
    'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
    't', 'u']

    smi2index = dict((c, i) for i, c in enumerate(SMILES_CHARS))
    index2smi = dict((i, c) for i, c in enumerate(SMILES_CHARS))
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    X = np.zeros((maxlen, len(SMILES_CHARS)))
    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1
    return X
    

def smiles_decoder(X):
    smi = ''
    X = X.argmax(axis=-1)
    for i in X:
        smi += index2smi[i]
    return smi


def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    return npfp

def maccs_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    npfp = np.array(list(fp.ToBitString())).astype('int8')
    return npfp

if __name__ == "__main__":

    train_set = pd.read_csv("data/train.csv")
    # train_set = pd.read_csv("data/augmentation/train_aug_random20.csv")
    train_set, train_label = list(train_set['SMILES']), list(train_set['SENTENCE'])
    train_label = [i.split(',') for i in train_label]

    test_set = pd.read_csv("data/test.csv")
    test_set = list(test_set['SMILES'])

    vocab = open("data/vocabulary.txt", 'r').read().split("\n")

    # count the number of occurences for each label
    # train_label_sub = [item for sublist in train_label for item in sublist]
    # counts = dict((x, train_label_sub.count(x)) for x in set(train_label_sub))
    # pprint(counts)

    # Morgan encoding
    print(f"Size of train set: {len(train_set)}")
    fingerprint = np.array([morgan_fp(i) for i in train_set])
    np.savez_compressed("data/train_set_fingerprint_2048bits_radius2.npz", morgan=fingerprint)
    print("NumPy compressed file has been saved!")

    fingerprint = np.array([morgan_fp(i) for i in test_set])
    np.savez_compressed("data/test_set_fingerprint_2048bits_radius2.npz", morgan=fingerprint)
    print("NumPy compressed file has been saved!")

    exit(0)
