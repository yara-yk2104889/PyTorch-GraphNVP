"""
Compare generation metrics across temperatures for a saved GraphNVP checkpoint.

Example:
    python compare_temps.py --data_name qm9 --model_save_dir ./saved_qm9_models --epoch 10
    python compare_temps.py --data_name moses --model_save_dir ./saved_moses_models --epoch 10
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from os.path import join as joinpath

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.argparser import args as global_args
from graph_nvp.hyperparams import Hyperparameters
from graph_nvp.nvp_model import GraphNvpModel
from utils.data_utils import construct_mol
from utils import environment as env


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k', 'moses'])
    parser.add_argument('--model_save_dir', type=str, default='./saved_qm9_models')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--num_gen', type=int, default=100)
    parser.add_argument('--max_resample', type=int, default=300)
    parser.add_argument('--temps', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    return parser.parse_args()


def setup_model(data_name, model_save_dir, epoch):
    if data_name == 'qm9':
        global_args.atomic_num_list = [6, 7, 8, 9, 0]
        global_args.num_atoms = 9
        mlp_channels = [256, 256]
        gnn_channels = {'gcn': [8, 64], 'hidden': [64, 128]}
        num_masks = {'node': 9, 'channel': 9}
        mask_size = {'node': 15, 'channel': -1}
        num_coupling = {'node': 18, 'channel': 14}
    elif data_name == 'zinc250k':
        global_args.atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        global_args.num_atoms = 38
        mlp_channels = [1024, 512]
        gnn_channels = {'gcn': [16, 128], 'hidden': [64, 256]}
        num_masks = {'node': 19, 'channel': 19}
        mask_size = {'node': 15, 'channel': -1}
        num_coupling = {'node': 18, 'channel': 14}
    elif data_name == 'moses':
        global_args.atomic_num_list = [6, 7, 8, 9, 16, 17, 35, 0]
        global_args.num_atoms = 26
        mlp_channels = [1024, 512]
        gnn_channels = {'gcn': [16, 128], 'hidden': [64, 256]}
        num_masks = {'node': 13, 'channel': 13}
        mask_size = {'node': 15, 'channel': -1}
        num_coupling = {'node': 18, 'channel': 14}

    num_atoms = global_args.num_atoms
    num_rels = 4
    num_features = len(global_args.atomic_num_list)

    params = Hyperparameters(
        num_atoms, num_rels, num_features,
        num_masks=num_masks, mask_size=mask_size, num_coupling=num_coupling,
        batch_norm=True, additive_transformations=True,
        mlp_channels=mlp_channels, gnn_channels=gnn_channels,
    )
    model = GraphNvpModel(params).to(global_args.device)
    ckpt = joinpath(model_save_dir, f'epoch{epoch}-mle.ckpt')
    model.load_state_dict(torch.load(ckpt, map_location=global_args.device))
    model.eval()
    print(f'Loaded checkpoint: {ckpt}')
    return model


def sample_z(model, temp):
    z_dim = model.adj_size + model.x_size
    mu = np.zeros(z_dim, dtype=np.float32)
    sigma = temp * np.sqrt(np.exp(model.prior_ln_var.item())) * np.ones(z_dim)
    z = np.random.normal(mu, sigma, (1, z_dim)).astype(np.float32)
    return torch.from_numpy(z).float().to(global_args.device)


def generate_at_temp(model, temp, num_gen, max_resample, all_train_smiles):
    all_smiles, pure_valids = [], []
    appear_in_train = 0
    cnt_gen = 0

    while len(all_smiles) < num_gen and cnt_gen < max_resample:
        z = sample_z(model, temp)
        A, X = model.reverse(z, model.x_size)
        X = F.softmax(X, dim=2)
        mols = [construct_mol(x_elem, adj_elem, global_args.atomic_num_list)
                for x_elem, adj_elem in zip(X, A)]
        cnt_gen += 1
        num_atoms = -1
        smiles = ''
        for mol in mols:
            if mol is None:
                continue
            if not env.check_chemical_validity(mol) or not env.check_valency(mol):
                continue
            num_atoms = mol.GetNumAtoms()
            smiles = Chem.MolToSmiles(mol)
        if num_atoms < 2 or '.' in smiles:
            continue
        all_smiles.append(smiles)
        pure_valids.append(1.0)
        if all_train_smiles and smiles in all_train_smiles:
            appear_in_train += 1

    n = num_gen
    valid_rate = len(all_smiles) / n
    unique_rate = len(set(all_smiles)) / max(len(all_smiles), 1)
    novelty = 1.0 - (appear_in_train / max(len(all_smiles), 1))
    return valid_rate, unique_rate, novelty, len(all_smiles)


def main():
    opts = parse()
    global_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = setup_model(opts.data_name, opts.model_save_dir, opts.epoch)

    txt_path = f'./data/{opts.data_name}_kekulized_ggnp.txt'
    try:
        with open(txt_path) as f:
            train_smiles = set(s.strip() for s in f)
    except FileNotFoundError:
        train_smiles = None
        print(f'Warning: could not load training SMILES from {txt_path}, novelty will be 0')

    print(f'\nGenerating {opts.num_gen} molecules per temperature...\n')
    print(f"{'Temp':>6}  {'Valid%':>8}  {'Unique%':>8}  {'Novelty%':>9}  {'#Generated':>10}")
    print('-' * 50)

    for temp in opts.temps:
        valid, unique, novelty, n_gen = generate_at_temp(
            model, temp, opts.num_gen, opts.max_resample, train_smiles
        )
        print(f'{temp:>6.2f}  {valid*100:>8.2f}  {unique*100:>8.2f}  {novelty*100:>9.2f}  {n_gen:>10}')


if __name__ == '__main__':
    main()
