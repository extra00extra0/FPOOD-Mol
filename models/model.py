import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np

from vector_quantize_pytorch import VectorQuantize
# from .vq_update import VectorQuantize

from .gnnconv import GNN_node
from .pooling import MorganFingerprintPool


FP_DIM_DICT = {
    'MorganFP': 1024,
    'RdkitFP': 1024,
    'EstateFP': 79,
    'MACCSFP': 166,
    'PubChemFP': 881,
    'RGroupFP': 2048,
    'FragmentFP': 86,
}


class FeatureSeparator(nn.Module):
    def __init__(self, args, config, with_gnn=True, input_dim=None):
        super(FeatureSeparator, self).__init__()
        emb_d = args.emb_dim if input_dim is None else input_dim
        self.with_gnn = with_gnn
        if with_gnn:
            self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                                  drop_ratio=args.dropout, gnn_type=args.gnn_type)
        else:
            self.r_gnn = None

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       nn.LayerNorm(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, 1),
                                       nn.Sigmoid())
        self.args = args

    def forward(self, data_or_feat, batch=None):
        if self.with_gnn:
            feat = self.r_gnn(data_or_feat)
            batch = data_or_feat.batch
        else:
            feat = data_or_feat

        score = self.separator(feat)
        if batch is None:
            return score

        # reg on score
        pos_score_on_node = score.squeeze(-1)  # [n]
        pos_score_on_batch = scatter_add(pos_score_on_node, batch, dim=0)  # [B]
        neg_score_on_batch = scatter_add((1 - pos_score_on_node), batch, dim=0)  # [B]
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8


class DiscreteEncoder(nn.Module):
    def __init__(self, args, config):
        super(DiscreteEncoder, self).__init__()
        self.args = args
        self.config = config
        self.fp_dim = getattr(args, 'fp_dim', FP_DIM_DICT.get(args.fp_type, 1024))
        self.fp_score_mode = args.fp_score_mode
        self.logit_fusion_mode = args.logit_fusion_mode
        self.prediction_mode = args.prediction_mode
        emb_dim = args.emb_dim
        out_dim = getattr(getattr(config, 'dataset', None), 'num_classes', 1)
        self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                            drop_ratio=args.dropout, gnn_type=args.gnn_type)
        self.atom_separator = FeatureSeparator(args, config, with_gnn=False)
        self.fp_pool = MorganFingerprintPool(in_channel=emb_dim, fp_dim=self.fp_dim,
                                             pool_type=args.fp_pool_type,
                                             num_heads=args.fp_attn_heads)
        if self.fp_score_mode == 'graph_fp_pretrain':
            self.fp_separator = FeatureSeparator(args, config, with_gnn=False, input_dim=1)
            self.fp_pretrain_head = nn.Linear(self.fp_dim, out_dim)
        else:
            self.fp_separator = FeatureSeparator(args, config, with_gnn=False)
            self.fp_pretrain_head = None
        self.graph_pool = global_mean_pool
        self.atom_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                             nn.BatchNorm1d(emb_dim * 2),
                                             nn.ReLU(),
                                             nn.Dropout(),
                                             nn.Linear(emb_dim * 2, out_dim))
        self.fp_classifier = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                           nn.BatchNorm1d(emb_dim * 2),
                                           nn.ReLU(),
                                           nn.Dropout(),
                                           nn.Linear(emb_dim * 2, out_dim))
        self.logit_fusion = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 2),
                                          nn.BatchNorm1d(emb_dim * 2),
                                          nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(emb_dim * 2, 2))

        self.vq = VectorQuantize(dim=emb_dim,
                                 codebook_size=args.num_e,
                                 commitment_weight=args.commitment_weight,
                                 decay=0.9)

        self.mix_proj = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim),
                                      nn.BatchNorm1d(emb_dim),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(emb_dim, emb_dim))

        self.simsiam_proj = nn.Sequential(nn.Linear(emb_dim, emb_dim * 2),
                                          nn.BatchNorm1d(emb_dim * 2),
                                          nn.ReLU(),
                                          nn.Linear(emb_dim * 2, emb_dim))

    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def _get_graph_fp(self, data):
        graph_fp = scatter_add(data.fp.float(), data.batch, dim=0)
        graph_fp = (graph_fp > 0).float()
        return graph_fp

    def pretrain_fp(self, data):
        if self.fp_score_mode != 'graph_fp_pretrain':
            raise RuntimeError('fp pretraining is only available when fp_score_mode=graph_fp_pretrain')
        graph_fp = self._get_graph_fp(data)
        fp_score = self.fp_separator(graph_fp.unsqueeze(-1))
        weighted_fp = (graph_fp.unsqueeze(-1) * fp_score).squeeze(-1)
        return self.fp_pretrain_head(weighted_fp)

    def forward(self, data):
        node_feat = self.gnn(data)
        atom_graph_score = self.atom_separator(node_feat)
        atom_graph_feat = node_feat * atom_graph_score
        atom_graph_repr = None
        atom_logit = None
        if self.prediction_mode == 'fusion':
            atom_graph_repr = global_add_pool(atom_graph_feat, data.batch)
            atom_logit = self.atom_classifier(atom_graph_repr)

        cmt_loss = torch.zeros((), device=node_feat.device)
        if hasattr(data, 'fp') and data.fp is not None:
            fp_bit_repr = self.fp_pool(atom_graph_feat, data.batch, data.fp)
            if self.args.use_vq_residual:
                fp_bit_vq, _, cmt_loss = self.vq(fp_bit_repr)
                fp_bit_repr = (fp_bit_repr + fp_bit_vq) / 2

            if self.fp_score_mode == 'graph_fp_pretrain':
                graph_fp = self._get_graph_fp(data)
                c_graph_score = self.fp_separator(graph_fp.unsqueeze(-1))
            else:
                c_graph_score = self.fp_separator(fp_bit_repr)
            c_graph_feat = fp_bit_repr * c_graph_score
            fp_graph_repr = c_graph_feat.sum(dim=1)
            fp_logit = self.fp_classifier(fp_graph_repr)
            if self.prediction_mode == 'fp_only':
                c_logit = fp_logit
            elif self.logit_fusion_mode == 'rescale':
                fusion_weight = self.logit_fusion(torch.cat([atom_graph_repr, fp_graph_repr], dim=-1))
                fusion_weight = torch.softmax(fusion_weight, dim=-1)
                c_logit = atom_logit * fusion_weight[:, :1] + fp_logit * fusion_weight[:, 1:2]
            else:
                c_logit = (atom_logit + fp_logit) / 2
        else:
            c_graph_feat = self.graph_pool(atom_graph_feat, data.batch)
            if self.prediction_mode == 'fp_only':
                c_logit = self.fp_classifier(c_graph_feat)
            else:
                c_logit = atom_logit

        return c_logit, c_graph_feat, atom_graph_feat, cmt_loss


class MyModel(nn.Module):
    def __init__(self, args, config):
        super(MyModel, self).__init__()
        self.args = args
        self.config = config

        self.encoder = DiscreteEncoder(args, config)

    def forward(self, data):
        c_logit, c_graph_feat, s_graph_feat, cmt_loss = self.encoder(data)
        return c_logit, cmt_loss

    def pretrain_fp(self, data):
        return self.encoder.pretrain_fp(data)

    def mix_cs_proj(self, c_f: torch.Tensor, s_f: torch.Tensor):
        n = c_f.size(0)
        perm = np.random.permutation(n)
        mix_f = torch.cat([c_f, s_f[perm]], dim=-1)
        proj_mix_f = self.encoder.mix_proj(mix_f)
        return proj_mix_f
