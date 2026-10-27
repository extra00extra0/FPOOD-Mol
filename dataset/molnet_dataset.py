import csv
import os
import os.path as osp
import pickle
import random
import urllib.request
from collections import defaultdict

import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .smiles2graph import smile2graph4drugood_fppool


DATASET_META = {
    'BBBP': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',
        'target_col': 'p_np',
        'smiles_col': 'smiles',
        'task_type': 'Binary classification',
        'metric': 'ROC-AUC',
    },
    'BACE': {
        'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
        'target_col': 'Class',
        'smiles_col': 'mol',
        'task_type': 'Binary classification',
        'metric': 'ROC-AUC',
    },
}


def _safe_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split_indices(smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1):
    total = frac_train + frac_valid + frac_test
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f'Split fractions must sum to 1, got {total}')

    scaffold_to_indices = defaultdict(list)
    for index, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffold_to_indices[scaffold].append(index)

    scaffold_sets = sorted(
        scaffold_to_indices.values(),
        key=lambda index_set: (-len(index_set), index_set[0])
    )

    num_total = len(smiles_list)
    train_cutoff = frac_train * num_total
    valid_cutoff = (frac_train + frac_valid) * num_total

    train_groups, valid_groups, test_groups = [], [], []
    for index_set in scaffold_sets:
        train_size = sum(len(group) for group in train_groups)
        valid_size = sum(len(group) for group in valid_groups)
        if train_size + len(index_set) <= train_cutoff:
            train_groups.append(index_set)
        elif train_size + valid_size + len(index_set) <= valid_cutoff:
            valid_groups.append(index_set)
        else:
            test_groups.append(index_set)

    if len(valid_groups) == 0 and len(test_groups) > 1:
        valid_groups.append(test_groups.pop())
    if len(test_groups) == 0 and len(valid_groups) > 1:
        test_groups.append(valid_groups.pop())

    train_idx = [idx for group in train_groups for idx in group]
    valid_idx = [idx for group in valid_groups for idx in group]
    test_idx = [idx for group in test_groups for idx in group]

    return train_idx, valid_idx, test_idx


class MoleculeScaffoldDataset(InMemoryDataset):
    def __init__(self, name, root='data', transform=None, pre_transform=None, pre_filter=None,
                 fp_type='MorganFP', split_seed=0):
        if name not in DATASET_META:
            raise ValueError(f'Unsupported MoleculeNet dataset: {name}')

        self.name = name
        self.root = root
        self.fp_type = fp_type
        self.split_seed = split_seed
        self.meta = DATASET_META[name]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = _safe_torch_load(self.processed_paths[0])
        with open(self.processed_paths[1], 'rb') as fin:
            self.data_cfg = pickle.load(fin)
        with open(self.processed_paths[2], 'rb') as fin:
            self.data_statistics = pickle.load(fin)
        with open(self.processed_paths[3], 'rb') as fin:
            self.train_index, self.valid_index, self.test_index = pickle.load(fin)
        self.num_tasks = 1

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return [f'{self.name}.csv']

    @property
    def processed_dir(self):
        fp_tag = self.fp_type.lower().replace('fp', '')
        return osp.join(
            self.root,
            self.name,
            f'processed-scaffold-fppool-{fp_tag}-seed{self.split_seed}'
        )

    @property
    def processed_file_names(self):
        return 'data.pt', 'cfg.pt', 'statistics.pt', 'split.pt'

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        target_path = self.raw_paths[0]
        if osp.exists(target_path):
            return
        urllib.request.urlretrieve(self.meta['url'], target_path)

    def _read_rows(self):
        rows = []
        with open(self.raw_paths[0], 'r', encoding='utf-8') as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                smiles = row.get(self.meta['smiles_col'])
                label = row.get(self.meta['target_col'])
                if not smiles or label in (None, ''):
                    continue
                try:
                    label = int(float(label))
                except ValueError:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                rows.append({
                    'smiles': smiles,
                    'label': label,
                })
        if not rows:
            raise RuntimeError(f'No valid molecules found in {self.raw_paths[0]}')
        return rows

    def process(self):
        rows = self._read_rows()
        rng = random.Random(self.split_seed)
        rng.shuffle(rows)

        data_list = []
        smiles_list = []
        labels = []
        for row in tqdm(rows, desc=f'Processing {self.name}'):
            smiles = row['smiles']
            x, edge_index, edge_attr, fp, fp_length = smile2graph4drugood_fppool(
                smiles,
                fp_type=self.fp_type
            )
            y = torch.tensor([[row['label']]], dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                fp=fp,
                fp_length=fp_length,
                y=y,
                smiles=smiles,
            )
            data.batch_num_nodes = data.num_nodes
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
            smiles_list.append(smiles)
            labels.append(row['label'])

        train_index, valid_index, test_index = scaffold_split_indices(smiles_list)

        statistics = {
            'num_samples': len(data_list),
            'num_train': len(train_index),
            'num_valid': len(valid_index),
            'num_test': len(test_index),
            'positive_ratio': float(sum(labels) / len(labels)),
        }
        cfg = {
            'dataset_name': self.name,
            'split_type': 'scaffold',
            'task_type': self.meta['task_type'],
            'metric': self.meta['metric'],
            'fp_type': self.fp_type,
        }

        torch.save(self.collate(data_list), self.processed_paths[0])
        with open(self.processed_paths[1], 'wb') as fout:
            pickle.dump(cfg, fout)
        with open(self.processed_paths[2], 'wb') as fout:
            pickle.dump(statistics, fout)
        with open(self.processed_paths[3], 'wb') as fout:
            pickle.dump([train_index, valid_index, test_index], fout)

    def __repr__(self):
        return f'{self.name}({len(self)})'
