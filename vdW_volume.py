import pybel

# conda install -c openbabel openbabel

def get_vdw(smiles):
    '''
    This function returns vdW of a molecule using a formula in this paper:
    https://pubs.acs.org/doi/pdf/10.1021/jo034808o
    '''

    # reading SMILES
    mol = pybel.readstring("smi", smiles)
    mol.OBMol.AddHydrogens()

    no_atoms = len(mol.atoms)
    bonds = mol.OBMol.NumBonds()
    no_ar = 0
    no_non_ar = 0

    # calculate the no of aromatic and non-aromatic rings
    for r in mol.OBMol.GetSSSR():
        if r.IsAromatic():
            no_ar = no_ar+1
        else:
            no_non_ar = no_non_ar+1

    def get_num_struc(smarts):
        smarts = pybel.Smarts(smarts)
        num_unique_matches = len(smarts.findall(mol))
        return num_unique_matches

    # Calculating no.of fusions
    no_f_ring_AlAr = get_num_struc('[A]~@[*](~@[a])~@[*](~@[a])~@[A]')
    no_f_ring_AlAl = get_num_struc('[A]~@[*](~@[A])~@[*](~@[A])~@[A]')
    no_f_ring_ArAr = get_num_struc('[a]~@[*](~@[a])~@[*](~@[a])~@[a]')
    no_f_ring_S = get_num_struc('[s]1~@[c](~@[c])~@[c](~@[c])~@[c](~@[c])~@[c]1(~@[c])')

    # Calculating no.of atoms
    no_of_C = smiles.count('c') + smiles.count('C')
    no_of_N = smiles.count('n') + smiles.count('N')
    no_of_O = smiles.count('o') + smiles.count('O')
    no_of_F = smiles.count('f') + smiles.count('F')
    no_of_Cl = smiles.count('Cl')
    no_of_Br = smiles.count('Br')
    no_of_I = smiles.count('i') + smiles.count('I')
    no_of_P = smiles.count('p') + smiles.count('P')
    no_of_S = smiles.count('s') + smiles.count('S')
    no_of_Si = smiles.count('si') + smiles.count('Si')
    no_of_Se = smiles.count('se') + smiles.count('Se')
    no_of_Te = smiles.count('te') + smiles.count('Te')

    no_of_H = no_atoms - (no_of_C + no_of_N + no_of_O + no_of_F + no_of_Cl +
                          no_of_Br + no_of_I + no_of_P + no_of_S + no_of_Si + no_of_Se + no_of_Te)

    no_of_C = no_of_C - no_of_Cl
    no_of_S = no_of_S - no_of_Si - no_of_Se

    V_vdw = (no_of_H)*7.24 + \
        (no_of_C)*20.58 + \
        (no_of_N)*15.6 + \
        (no_of_O)*14.71 + \
        (no_of_F)*13.31 + \
        (no_of_Cl)*22.45 + \
        (no_of_Br)*26.52 + \
        (no_of_I)*32.52 + \
        (no_of_P)*24.43 + \
        (no_of_S)*24.43 + \
        (no_of_Si)*38.79 + \
        (no_of_Se)*28.73 + \
        (no_of_Te)*36.62

    V_vdw = V_vdw - 5.92*(bonds) \
        - 14.7*(no_ar) \
        - 3.8*(no_non_ar) \
        + 5*(no_f_ring_ArAr) \
        + 3*(no_f_ring_AlAr) \
        + 1*(no_f_ring_AlAl) \
        - 5*(no_f_ring_S)

    return V_vdw
