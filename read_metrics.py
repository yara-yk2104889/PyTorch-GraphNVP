import numpy as np
import glob
import os
import re

def read_metrics(gen_path='./results/qm9_mols'):
    pattern = re.compile(r'valid(\d+)\.npy')
    epochs = []
    for f in glob.glob(os.path.join(gen_path, 'valid*.npy')):
        m = pattern.match(os.path.basename(f))
        if m:
            epochs.append(int(m.group(1)))
    epochs.sort()

    if not epochs:
        print(f'No metric files found in {gen_path}')
        return

    print(f"{'Epoch':>6}  {'Validity':>10}  {'Uniqueness':>10}  {'Novelty':>10}")
    print('-' * 44)
    for epoch in epochs:
        valid    = np.load(os.path.join(gen_path, f'valid{epoch}.npy'))
        unique   = np.load(os.path.join(gen_path, f'unique{epoch}.npy'))
        novelty  = np.load(os.path.join(gen_path, f'novelty{epoch}.npy'))
        # each file stores cumulative history; take the last value
        print(f'{epoch:>6}  {valid[-1]:>10.4f}  {unique[-1]:>10.4f}  {novelty[-1]:>10.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', default='./results/qm9_mols')
    args = parser.parse_args()
    read_metrics(args.gen_path)
