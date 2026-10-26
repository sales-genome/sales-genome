"""Complete captured graph-variant builders; no file-layout assumptions."""
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData

def build_bipartite(data, train_vc):
    g = HeteroData()

    offers_ei = data["vendor", "offers", "product"].edge_index
    product_x = data["product"].x
    n_vendors = data["vendor"].num_nodes

    vendor_feat = torch.zeros(n_vendors, product_x.shape[1])
    counts = torch.zeros(n_vendors)
    for i in range(offers_ei.shape[1]):
        v, p = offers_ei[0, i].item(), offers_ei[1, i].item()
        vendor_feat[v] += product_x[p]
        counts[v] += 1
    mask = counts > 0
    vendor_feat[mask] /= counts[mask].unsqueeze(1)

    g["vendor"].x = vendor_feat
    g["customer"].x = data["customer"].x.clone()

    src = torch.tensor([v for v, c in train_vc])
    dst = torch.tensor([c for v, c in train_vc])
    g["vendor", "has_case_study", "customer"].edge_index = torch.stack([src, dst])
    g["customer", "rev_has_case_study", "vendor"].edge_index = torch.stack([dst, src])
    return g

def build_collapsed(data, train_vc, add_residual=False):
    g = HeteroData()

    g["vendor_product"].x = data["product"].x.clone()
    g["cap_canonical"].x = data["cap_canonical"].x.clone()
    g["pain_canonical"].x = data["pain_canonical"].x.clone()
    g["customer"].x = data["customer"].x.clone()

    for src_t, rel, dst_t, orig_et in [
        ("vendor_product", "provides", "cap_canonical",
         ("product", "provides", "cap_canonical")),
        ("cap_canonical", "addresses", "pain_canonical",
         ("cap_canonical", "addresses", "pain_canonical")),
        ("customer", "experiences", "pain_canonical",
         ("customer", "experiences", "pain_canonical")),
    ]:
        ei = data[orig_et].edge_index
        g[src_t, rel, dst_t].edge_index = ei
        g[dst_t, f"rev_{rel}", src_t].edge_index = ei.flip(0)

    if add_residual:
        offers_ei = data["vendor", "offers", "product"].edge_index
        vendor_products = defaultdict(list)
        for i in range(offers_ei.shape[1]):
            vendor_products[offers_ei[0, i].item()].append(offers_ei[1, i].item())

        vp_src, vp_dst = [], []
        for v, c in train_vc:
            for p in vendor_products.get(v, []):
                vp_src.append(p)
                vp_dst.append(c)

        if vp_src:
            s, d = torch.tensor(vp_src), torch.tensor(vp_dst)
            g["vendor_product", "case_study", "customer"].edge_index = torch.stack([s, d])
            g["customer", "rev_case_study", "vendor_product"].edge_index = torch.stack([d, s])

    offers_ei = data["vendor", "offers", "product"].edge_index
    v2p = defaultdict(list)
    for i in range(offers_ei.shape[1]):
        v2p[offers_ei[0, i].item()].append(offers_ei[1, i].item())
    return g, dict(v2p)

def build_full_reverse(data):
    g = HeteroData()
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            g[nt].x = data[nt].x.clone()
        else:
            g[nt].num_nodes = data[nt].num_nodes

    for et in data.edge_types:
        ei = data[et].edge_index
        src_t, rel, dst_t = et
        g[et].edge_index = ei.clone()
        g[(dst_t, f"rev_{rel}", src_t)].edge_index = ei.flip(0)
    return g
