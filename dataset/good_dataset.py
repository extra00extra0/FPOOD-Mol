import os.path as osp
import pickle

import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from .smiles2graph import smile2graph4drugood_fppool


LOCAL_GOOD_DATASETS = {'GOODZINC'}

GOOD_TASK_META = {
    'GOODZINC': {
        'task_type': 'Regression',
        'metric': 'MAE',
    },
}

OPTIONAL_GRAPH_ATTRS = ('domain_id', 'env_id', 'pyx', 'idx')


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_data_object(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise RuntimeError('Loaded GOOD object is empty.')
        return obj[0]
    return obj


def _ensure_tensor(value):
    if value is None or torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


class LocalGOODDataset(InMemoryDataset):
    def __init__(self, name, root='data', domain='scaffold', shift='concept',
                 transform=None, pre_transform=None, pre_filter=None, fp_type='MorganFP'):
        if name not in LOCAL_GOOD_DATASETS:
            raise ValueError(f'Unsupported local GOOD dataset: {name}')

        self.name = name
        self.root = root
        self.domain = domain
        self.shift = shift
        self.fp_type = fp_type
        self.meta = GOOD_TASK_META[name]
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
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def raw_file_names(self):
        return [
            f'{self.shift}_train.pt',
            f'{self.shift}_val.pt',
            f'{self.shift}_test.pt',
        ]

    @property
    def processed_dir(self):
        fp_tag = self.fp_type.lower().replace('fp', '')
        return osp.join(
            self.root,
            f'{self.name}-local',
            self.domain,
            f'{self.shift}-fppool-{fp_tag}',
        )

    @property
    def processed_file_names(self):
        return 'data.pt', 'cfg.pt', 'statistics.pt', 'split.pt'

    def download(self):
        return

    def _collect_optional_attr_specs(self):
        attr_specs = {}
        for raw_path in self.raw_paths:
            raw_obj = _safe_torch_load(raw_path)
            data_obj = _extract_data_object(raw_obj)
            for attr_name in OPTIONAL_GRAPH_ATTRS:
                attr_value = _ensure_tensor(getattr(data_obj, attr_name, None))
                if attr_value is None:
                    continue
                attr_specs[attr_name] = {
                    'dtype': attr_value.dtype,
                    'fill_value': float('nan') if torch.is_floating_point(attr_value) else -1,
                }
        return attr_specs

    def _build_split(self, raw_path, optional_attr_specs):
        raw_obj = _safe_torch_load(raw_path)
        data_obj = _extract_data_object(raw_obj)

        smiles_list = list(data_obj.smiles)
        y_values = data_obj.y.view(-1).float()
        optional_values = {
            attr_name: _ensure_tensor(getattr(data_obj, attr_name, None))
            for attr_name in optional_attr_specs
        }

        split_data = []
        for i, smiles in enumerate(tqdm(smiles_list, desc=f'Converting {osp.basename(raw_path)}')):
            x, edge_index, edge_attr, fp, fp_length = smile2graph4drugood_fppool(
                smiles.strip(),
                fp_type=self.fp_type
            )
            item = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                fp=fp,
                fp_length=fp_length,
                y=y_values[i].view(1, 1),
                smiles=smiles.strip(),
            )
            for attr_name, attr_spec in optional_attr_specs.items():
                attr_value = optional_values[attr_name]
                if attr_value is None:
                    value = torch.full((1,), attr_spec['fill_value'], dtype=attr_spec['dtype'])
                else:
                    value = attr_value[i].view(1)
                setattr(item, attr_name, value)
            item.batch_num_nodes = item.num_nodes
            if self.pre_transform is not None:
                item = self.pre_transform(item)
            split_data.append(item)
        return split_data

    def process(self):
        optional_attr_specs = self._collect_optional_attr_specs()
        train_data = self._build_split(self.raw_paths[0], optional_attr_specs)
        valid_data = self._build_split(self.raw_paths[1], optional_attr_specs)
        test_data = self._build_split(self.raw_paths[2], optional_attr_specs)

        data_list = train_data + valid_data + test_data
        train_index = list(range(len(train_data)))
        valid_index = list(range(len(train_data), len(train_data) + len(valid_data)))
        test_index = list(range(len(train_data) + len(valid_data), len(data_list)))

        all_targets = torch.cat([data.y.view(-1) for data in data_list], dim=0)
        statistics = {
            'num_samples': len(data_list),
            'num_train': len(train_index),
            'num_valid': len(valid_index),
            'num_test': len(test_index),
            'target_mean': float(all_targets.mean().item()),
            'target_std': float(all_targets.std(unbiased=False).item()),
        }
        cfg = {
            'dataset_name': self.name,
            'domain': self.domain,
            'shift': self.shift,
            'task_type': self.meta['task_type'],
            'metric': self.meta['metric'],
            'fp_type': self.fp_type,
            'source_format': 'GOOD processed pt',
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
