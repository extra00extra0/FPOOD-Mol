import os
import logging
from tqdm import tqdm
from munch import Munch, munchify

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np
import warnings

from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, describe_model, save_model, load_model
from local_metrics import build_simple_metric, compute_score, prepare_batch_target
from models import MyModel
from dataset import LOCAL_GOOD_DATASETS, build_non_good_dataset

logger = logging.getLogger()
warnings.filterwarnings("ignore", category=FutureWarning)


def load_good_runtime():
    from GOOD import register
    from GOOD.utils.config_reader import load_config
    from GOOD.utils.metric import Metric
    from GOOD.data.dataset_manager import read_meta_info
    from GOOD.utils.evaluation import eval_data_preprocess, eval_score
    from GOOD.utils.train import nan2zero_get_mask
    return register, load_config, Metric, read_meta_info, eval_data_preprocess, eval_score, nan2zero_get_mask


class Runner:
    def __init__(self, args, logger_path):
        self.args = args
        self.device = torch.device(f'cuda')
        self.use_good_runtime = args.dataset.startswith('GOOD') and args.dataset not in LOCAL_GOOD_DATASETS

        if self.use_good_runtime:
            # for GOOD, load Config
            register, load_config, Metric, read_meta_info, _, _, _ = load_good_runtime()
            cfg_path = os.path.join(args.config_path, args.dataset, args.domain, args.shift, 'base.yaml')
            cfg, _, _ = load_config(path=cfg_path)
            cfg = munchify(cfg)
            cfg.device = self.device
            dataset, meta_info = register.datasets[cfg.dataset.dataset_name].load(dataset_root=args.data_root,
                                                                                  domain=cfg.dataset.domain,
                                                                                  shift=cfg.dataset.shift_type,
                                                                                  generate=cfg.dataset.generate)
            read_meta_info(meta_info, cfg)
            # cfg.dropout
            # cfg.bs
            # update dropout & bs
            cfg.model.dropout_rate = args.dropout
            cfg.train.train_bs = args.bs
            cfg.random_seed = args.random_seed

            loader = register.dataloader[cfg.dataset.dataloader_name].setup(dataset, cfg)
            self.train_loader = loader['train']
            self.valid_loader = loader['val']
            self.test_loader = loader['test']

            self.metric = Metric()
            self.metric.set_score_func(dataset['metric'] if type(dataset) is dict else getattr(dataset, 'metric'))
            self.metric.set_loss_func(dataset['task'] if type(dataset) is dict else getattr(dataset, 'task'))
            cfg.metric = self.metric
        else:
            # DrugOOD / BBBP / BACE / local GOODZINC
            dataset = build_non_good_dataset(args)
            self.train_set = dataset[dataset.train_index]
            self.valid_set = dataset[dataset.valid_index]
            self.test_set = dataset[dataset.test_index]
            args.fp_dim = int(dataset[0].fp_length.view(-1)[0].item())

            self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True, drop_last=True)
            self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False)
            self.test_loader = DataLoader(self.test_set, batch_size=args.bs, shuffle=False)
            task_name = dataset.data_cfg.get('task_type', 'Binary classification')
            metric_name = dataset.data_cfg.get('metric', 'ROC-AUC')
            self.metric = build_simple_metric(task_name=task_name, metric_name=metric_name)
            cfg = Munch()
            cfg.metric = self.metric
            cfg.model = Munch()
            cfg.model.model_level = 'graph'
            cfg.task_name = task_name

        self.model = MyModel(args=args, config=cfg).to(self.device)
        self.model.load_state_dict(load_model(args.load_path, map_location=self.device))
        self.logger_path = logger_path

        self.cfg = cfg

    
    def run(self):
        train_score = self.test_step(self.train_loader)
        val_score = self.test_step(self.valid_loader)
        test_score = self.test_step(self.test_loader)
        logger.info(f"TRAIN={train_score:.5f}, VAL={val_score:.5f}, TEST={test_score:.5f}")


    @torch.no_grad()
    def test_step(self, loader):

        self.model.eval()
        y_pred, y_gt = [], []
        for data in loader:
            data = data.to(self.device)
            logit, _ = self.model(data)
            if self.use_good_runtime:
                _, _, _, _, eval_data_preprocess, eval_score, nan2zero_get_mask = load_good_runtime()
                mask, _ = nan2zero_get_mask(data, 'None', self.cfg)
                pred, target = eval_data_preprocess(data.y, logit, mask, self.cfg)
            else:
                mask, target = prepare_batch_target(data, self.cfg.task_name)
                pred = logit[mask.bool()].view(-1, 1)
                target = target[mask.bool()].view(-1, 1)
            y_pred.append(pred)
            y_gt.append(target)

        if self.use_good_runtime:
            score = eval_score(y_pred, y_gt, self.cfg)
        else:
            score = compute_score(self.metric, y_pred, y_gt)

        return score

def main():
    args = args_parser()
    torch.cuda.set_device(int(args.gpu))
    if isinstance(args.random_seed, list):
        args.random_seed = args.random_seed[0]

    logger = initialize_exp(args)
    set_seed(args.random_seed)
    logger_path = get_dump_path(args)

    runner = Runner(args, logger_path)
    runner.run()


if __name__ == '__main__':
    main()
