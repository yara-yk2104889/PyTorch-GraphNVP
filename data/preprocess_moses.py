"""
Preprocess moses_70k_smiles.txt into the .npz format expected by GraphNVP.
Run from the project root:
    python data/preprocess_moses.py
"""
import json
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from chainer_chemistry.datasets import NumpyTupleDataset

MAX_ATOMS = 26
DATA_DIR = './data'
SMILES_FILE = os.path.join(DATA_DIR, 'moses', 'moses_70k_smiles.txt')
OUT_NPZ = os.path.join(DATA_DIR, 'moses_relgcn_kekulized_ggnp.npz')
OUT_TXT = os.path.join(DATA_DIR, 'moses_kekulized_ggnp.txt')
OUT_CFG = os.path.join(DATA_DIR, 'moses_config.txt')
OUT_VAL = os.path.join(DATA_DIR, 'valid_idx_moses.json')

ATOM_LIST = [6, 7, 8, 9, 16, 17, 35, 0]  # C N O F S Cl Br + padding

BOND_TYPE_TO_IDX = {
    Chem.rdchem.BondType.SINGLE:   0,
    Chem.rdchem.BondType.DOUBLE:   1,
    Chem.rdchem.BondType.TRIPLE:   2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

def mol_to_arrays(mol, max_atoms):
    """Convert an RDKit mol to (atom_array, adj_array) matching GGNNPreprocessor output."""
    # Kekulize to remove aromatic bonds
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        return None, None

    n = mol.GetNumAtoms()
    if n > max_atoms:
        return None, None

    # Atom array: atomic numbers, padded with 0
    atom_array = np.zeros(max_atoms, dtype=np.int32)
    for i, atom in enumerate(mol.GetAtoms()):
        atom_array[i] = atom.GetAtomicNum()

    # Adj array: shape [4, max_atoms, max_atoms], one channel per bond type
    adj_array = np.zeros((4, max_atoms, max_atoms), dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        ch = BOND_TYPE_TO_IDX.get(bt, 3)
        adj_array[ch, i, j] = 1.0
        adj_array[ch, j, i] = 1.0

    return atom_array, adj_array


nodes_list, adjs_list, labels_list, smiles_out = [], [], [], []

with open(SMILES_FILE) as f:
    smiles_all = [l.strip() for l in f if l.strip()]

print(f'Processing {len(smiles_all)} SMILES...')
for i, smi in enumerate(smiles_all):
    if i % 10000 == 0:
        print(f'  {i}/{len(smiles_all)}')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    atom_array, adj_array = mol_to_arrays(mol, MAX_ATOMS)
    if atom_array is None:
        continue
    nodes_list.append(atom_array)
    adjs_list.append(adj_array)
    labels_list.append(np.zeros(1, dtype=np.float32))
    smiles_out.append(smi)

print(f'Successfully processed {len(smiles_out)} molecules.')

dataset = NumpyTupleDataset(
    np.array(nodes_list),
    np.array(adjs_list),
    np.array(labels_list),
)
NumpyTupleDataset.save(OUT_NPZ, dataset)
print(f'Saved dataset to {OUT_NPZ}')

with open(OUT_TXT, 'w') as f:
    for smi in smiles_out:
        f.write(smi + '\n')
print(f'Saved SMILES to {OUT_TXT}')

config = {
    'atom_list': ATOM_LIST,
    'freedom': 0,
    'node_dim': len(ATOM_LIST),
    'max_size': MAX_ATOMS,
    'bond_dim': 4,
}
with open(OUT_CFG, 'w') as f:
    f.write(str(config))
print(f'Saved config to {OUT_CFG}')

n = len(smiles_out)
val_size = 5000
val_ids = list(range(n - val_size, n))
with open(OUT_VAL, 'w') as f:
    json.dump({'valid_idxs': [str(i + 1) for i in val_ids]}, f)
print(f'Saved val split ({val_size} mols) to {OUT_VAL}')
