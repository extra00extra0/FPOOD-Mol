import numpy as np
import torch
import torch.nn.functional as F


def _average_ranks(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def binary_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = _average_ranks(y_score)
    auc = (ranks[pos_mask].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class SimpleMetric:
    def __init__(self, task_name, metric_name):
        self.task_name = task_name
        self.metric_name = metric_name
        self.lower_better = 1 if metric_name.upper() in {'MAE', 'RMSE', 'MSE'} else -1

    def loss_func(self, pred, target, reduction='mean'):
        if self.task_name == 'Binary classification':
            return F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
        if self.task_name == 'Regression':
            return F.mse_loss(pred, target, reduction=reduction)
        raise ValueError(f'Unsupported task type: {self.task_name}')

    def score_func(self, y_true, y_pred):
        if self.metric_name == 'ROC-AUC':
            return binary_roc_auc_score(y_true, y_pred)
        if self.metric_name == 'MAE':
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            return float(np.mean(np.abs(y_true - y_pred)))
        raise ValueError(f'Unsupported metric: {self.metric_name}')


def build_simple_metric(task_name, metric_name):
    return SimpleMetric(task_name=task_name, metric_name=metric_name)


def prepare_batch_target(data, task_name):
    target = data.y.float()
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    mask = ~torch.isnan(target)
    if task_name == 'Binary classification':
        target = target.clamp(0, 1)
    return mask.float(), target


def compute_score(metric, preds, targets):
    y_pred = torch.cat(preds, dim=0).view(-1).detach().cpu().numpy()
    y_true = torch.cat(targets, dim=0).view(-1).detach().cpu().numpy()
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    return metric.score_func(y_true=y_true, y_pred=y_pred)
