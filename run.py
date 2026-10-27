import os
import logging
import copy
from tqdm import tqdm
from munch import Munch, munchify

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np

from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, describe_model, save_model, load_model
from local_metrics import build_simple_metric, compute_score, prepare_batch_target
from models import MyModel
from dataset import LOCAL_GOOD_DATASETS, build_non_good_dataset

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger()


def load_good_runtime():
    from GOOD import register
    from GOOD.utils.config_reader import load_config
    from GOOD.utils.metric import Metric
    from GOOD.data.dataset_manager import read_meta_info
    from GOOD.utils.evaluation import eval_data_preprocess, eval_score
    from GOOD.utils.train import nan2zero_get_mask
    return register, load_config, Metric, read_meta_info, eval_data_preprocess, eval_score, nan2zero_get_mask


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Runner:
    def __init__(self, args, writer, logger_path):
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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.pretrain_opt = None
        if args.fp_score_mode == 'graph_fp_pretrain':
            self.pretrain_opt = torch.optim.Adam(
                list(self.model.encoder.fp_separator.parameters()) +
                list(self.model.encoder.fp_pretrain_head.parameters()),
                lr=args.lr
            )
        self.has_fp = (not self.use_good_runtime) and args.fp_score_mode == 'graph_fp_pretrain'

        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path

        self.cfg = cfg

    def run(self):
        if self.metric.lower_better == 1:
            best_valid_score, best_test_score = float('inf'), float('inf')
        else:
            best_valid_score, best_test_score = -1, -1

        pretrain_epochs = self.args.epoch // 2
        if self.has_fp and pretrain_epochs > 0:
            for e in range(pretrain_epochs):
                self.pretrain_step(e)
        
        for e in range(self.args.epoch):
            self.train_step(e)
            valid_score = self.test_step(self.valid_loader)
            tqdm.write(f"Seed={self.args.random_seed}, E={e}, valid={valid_score:.5f}, test-score={best_test_score:.5f}")
            if e < self.args.best_score_warmup:
                continue
            # if valid_score > best_valid_score:
            if (valid_score > best_valid_score and self.metric.lower_better == -1) or \
                    (valid_score < best_valid_score and self.metric.lower_better == 1):
                test_score = self.test_step(self.test_loader)
                best_valid_score = valid_score
                best_test_score = test_score
                tqdm.write(f"Seed={self.args.random_seed}, UPDATE test-score={best_test_score:.5f}")

        return best_test_score

    def pretrain_step(self, epoch):
        self.model.train()
        set_requires_grad([self.model], requires_grad=False)
        set_requires_grad([self.model.encoder.fp_separator, self.model.encoder.fp_pretrain_head], requires_grad=True)

        pbar = tqdm(self.train_loader, desc=f"P [{epoch}]")

        for data in pbar:
            data = data.to(self.device)
            logit = self.model.pretrain_fp(data)
            if self.use_good_runtime:
                _, _, _, _, _, _, nan2zero_get_mask = load_good_runtime()
                mask, target = nan2zero_get_mask(data, 'None', self.cfg)
            else:
                mask, target = prepare_batch_target(data, self.cfg.task_name)
            loss = self.metric.loss_func(logit, target.float(), reduction='none') * mask
            loss = loss.sum() / mask.sum()

            self.pretrain_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.fp_separator.parameters()) +
                list(self.model.encoder.fp_pretrain_head.parameters()), 5
            )
            self.pretrain_opt.step()

            pbar.set_postfix_str(f"pretrain_loss={loss.item():.4f}")
            self.writer.add_scalar('pretrain_loss', loss.item(), self.total_step)
            self.total_step += 1

    def train_step(self, epoch):
        self.model.train()
        set_requires_grad([self.model.encoder], requires_grad=True)

        pbar = tqdm(self.train_loader, desc=f"E [{epoch}]")

        for data in pbar:
            data = data.to(self.device)
            c_logit, cmt_loss = self.model(data)
            # classification loss on c
            if self.use_good_runtime:
                _, _, _, _, _, _, nan2zero_get_mask = load_good_runtime()
                mask, target = nan2zero_get_mask(data, 'None', self.cfg)
            else:
                mask, target = prepare_batch_target(data, self.cfg.task_name)
            cls_loss = self.metric.loss_func(c_logit, target.float(), reduction='none') * mask
            cls_loss = cls_loss.sum() / mask.sum()

            loss = cls_loss + cmt_loss

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()

            pbar.set_postfix_str(f"loss={loss.item():.4f}")
            self.writer.add_scalar('loss', loss.item(), self.total_step)
            # self.writer.add_scalar('cls-loss', cls_loss.item(), self.total_step)
            # self.writer.add_scalar('cmt-loss', cmt_loss.item(), self.total_step)
            # self.writer.add_scalar('reg-loss', reg_loss.item(), self.total_step)

            self.total_step += 1


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

    def simsiam_loss(self, causal_rep, mix_rep):
        causal_rep = causal_rep.detach()
        causal_rep = F.normalize(causal_rep, dim=1)
        mix_rep = F.normalize(mix_rep, dim=1)
        return -(causal_rep * mix_rep).sum(dim=1).mean()


def main():
    args = args_parser()
    torch.cuda.set_device(int(args.gpu))

    logger = initialize_exp(args)
    logger_path = get_dump_path(args)
    seed_scores = []

    for seed in args.random_seed:
        logger.info(f"Start training with random_seed={seed}")
        seed_args = copy.deepcopy(args)
        seed_args.random_seed = seed
        set_seed(seed)

        seed_logger_path = os.path.join(logger_path, f'seed_{seed}')
        os.makedirs(seed_logger_path, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(seed_logger_path, 'tensorboard'))

        runner = Runner(seed_args, writer, seed_logger_path)
        test_score = runner.run()
        writer.close()

        seed_scores.append(test_score)
        logger.info(f"seed={seed}, final-test-score={test_score:.5f}")

    score_mean = float(np.mean(seed_scores))
    score_std = float(np.std(seed_scores))
    logger.info(f"all-seed-final-test-score mean+/-std = {score_mean:.5f}+/-{score_std:.5f}")


if __name__ == '__main__':
    main()
