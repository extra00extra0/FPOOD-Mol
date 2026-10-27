import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


class MorganFingerprintPool(nn.Module):
    """Morgan-only fingerprint pooling.

    Input:
        x:  [num_nodes, in_dim]
        batch: [num_nodes]
        fp: [num_nodes, 1024]

    Output:
        [batch_size, 1024, in_dim]
    """

    def __init__(self, in_channel, fp_dim=1024, pool_type='sum', num_heads=4):
        super().__init__()
        self.in_channel = in_channel
        self.fp_dim = fp_dim
        self.pool_type = pool_type
        self.num_heads = num_heads
        if self.pool_type == 'attn':
            if self.num_heads < 1:
                raise ValueError(f'num_heads must be positive, got {self.num_heads}')
            self.gate = nn.Linear(in_channel, self.num_heads)
            self.head_fuse = nn.Linear(self.num_heads * in_channel, in_channel)
        elif self.pool_type != 'sum':
            raise ValueError(f'Unsupported fp pool type: {self.pool_type}')

    def forward(self, x, batch, fp):
        if self.pool_type == 'attn':
            return self._attn_pool(x, batch, fp)
        return self._sum_pool(x, batch, fp)

    def _sum_pool(self, x, batch, fp):
        fp = fp.to(x.dtype)
        batch_size = int(batch.max().item() + 1)

        per_channel_pool = []
        for channel_idx in range(x.size(-1)):
            x_channel = x[:, channel_idx].unsqueeze(-1)
            x_channel = x_channel * fp
            pooled = scatter_add(x_channel, batch, dim=0, dim_size=batch_size)
            per_channel_pool.append(pooled)

        return torch.stack(per_channel_pool, dim=-1)

    def _attn_pool(self, x, batch, fp):
        fp = fp.bool()
        batch_size = int(batch.max().item() + 1)

        hit_index = fp.nonzero(as_tuple=False)
        if hit_index.numel() == 0:
            return x.new_zeros(batch_size, self.fp_dim, self.in_channel)

        node_index = hit_index[:, 0]
        bit_index = hit_index[:, 1]
        graph_index = batch[node_index]
        graph_bit_index = graph_index * self.fp_dim + bit_index

        hit_feat = x[node_index]
        hit_score = self.gate(hit_feat)
        hit_attn = softmax(hit_score, graph_bit_index)

        weighted_feat = hit_attn.unsqueeze(-1) * hit_feat.unsqueeze(1)
        pooled = scatter_add(
            weighted_feat,
            graph_bit_index,
            dim=0,
            dim_size=batch_size * self.fp_dim
        )
        pooled = pooled.view(batch_size, self.fp_dim, self.num_heads, self.in_channel)

        return self.head_fuse(pooled.reshape(batch_size, self.fp_dim, self.num_heads * self.in_channel))
