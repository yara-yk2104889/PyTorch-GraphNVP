import numpy as np
import json

ATOM_LIST = [6, 7, 8, 9, 16, 17, 35, 0]  # C N O F S Cl Br + padding
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_LIST)}

def one_hot(data, out_size=26, num_max_id=8):
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id))
    for i, atom_num in enumerate(data):
        idx = ATOM_TO_IDX.get(int(atom_num), num_max_id - 1)
        b[i, idx] = 1
    return b

def transform_fn(data):
    node, adj, label = data
    node = one_hot(node).astype(np.float32)
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label

def get_val_ids():
    file_path = 'data/valid_idx_moses.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [int(idx) - 1 for idx in data['valid_idxs']]
    return val_ids
