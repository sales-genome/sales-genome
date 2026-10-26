"""Captured historical R-GCN classes with an explicit collapsed-d100 alias."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

COLLAPSED_VARIANTS = {"collapsed_reverse", "collapsed_residual", "collapsed_res_d100"}

class FeatureProjector(nn.Module):
    def __init__(self, data, hidden_dim):
        super().__init__()
        self.projections = nn.ModuleDict()
        self.embeddings = nn.ModuleDict()
        for ntype in data.node_types:
            if hasattr(data[ntype], "x") and data[ntype].x is not None:
                self.projections[ntype] = nn.Linear(data[ntype].x.shape[1], hidden_dim)
            else:
                n = data[ntype].num_nodes
                if isinstance(n, torch.Tensor):
                    n = n.item()
                self.embeddings[ntype] = nn.Embedding(n, hidden_dim)

    def forward(self, data):
        device = next(self.parameters()).device
        x_dict = {}
        for ntype in data.node_types:
            if ntype in self.projections:
                x_dict[ntype] = self.projections[ntype](data[ntype].x)
            elif ntype in self.embeddings:
                n = data[ntype].num_nodes
                if isinstance(n, torch.Tensor):
                    n = n.item()
                x_dict[ntype] = self.embeddings[ntype](torch.arange(n, device=device))
        return x_dict

class RGCNEncoder(nn.Module):
    def __init__(self, hidden_dim, num_relations, num_layers=3, dropout=0.2, num_bases=8):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        bases = min(num_bases, num_relations)
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=bases))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_type)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h
        return x

class LinkPredModel(nn.Module):
    def __init__(self, variant, data, hidden_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.variant = variant
        self.projector = FeatureProjector(data, hidden_dim)
        num_rels = len(data.edge_types)
        self.encoder = RGCNEncoder(hidden_dim, num_rels, num_layers, dropout)

    def _to_homogeneous(self, x_dict, data):
        node_types = sorted(x_dict.keys())
        offsets = {}
        offset = 0
        xs = []
        for ntype in node_types:
            offsets[ntype] = offset
            xs.append(x_dict[ntype])
            offset += x_dict[ntype].shape[0]
        x = torch.cat(xs, dim=0)

        eis, ets = [], []
        for idx, et in enumerate(data.edge_types):
            src_t, _, dst_t = et
            ei = data[et].edge_index.clone()
            ei[0] += offsets[src_t]
            ei[1] += offsets[dst_t]
            eis.append(ei)
            ets.append(torch.full((ei.shape[1],), idx, dtype=torch.long, device=ei.device))
        return x, torch.cat(eis, dim=1), torch.cat(ets), offsets

    def _from_homogeneous(self, z, offsets):
        z_dict = {}
        ntypes = list(offsets.keys())
        for i, ntype in enumerate(ntypes):
            start = offsets[ntype]
            end = offsets[ntypes[i + 1]] if i + 1 < len(ntypes) else z.shape[0]
            z_dict[ntype] = z[start:end]
        return z_dict

    def encode(self, data):
        x_dict = self.projector(data)
        x, ei, et, offsets = self._to_homogeneous(x_dict, data)
        z = self.encoder(x, ei, et)
        return self._from_homogeneous(z, offsets)

    def get_vendor_customer(self, z_dict, vendor_product_lists=None, n_vendors=None):
        if self.variant in COLLAPSED_VARIANTS:
            z_vp = z_dict["vendor_product"]
            device, dim = z_vp.device, z_vp.shape[1]
            z_v = torch.zeros(n_vendors, dim, device=device)
            for v, products in enumerate(vendor_product_lists):
                if products:
                    z_v[v] = z_vp[products].mean(dim=0)
            return z_v, z_dict["customer"]
        return z_dict["vendor"], z_dict["customer"]
