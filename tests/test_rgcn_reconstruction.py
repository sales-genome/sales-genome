import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import HeteroData

from paper_submission.rgcn import (
    Configuration, collapsed_graph, info_nce, negative_candidates, sampled_recovery, train,
)


def base_fixture():
    data = HeteroData()
    data["vendor"].num_nodes = 2
    for kind, count in [("product", 2), ("cap_canonical", 2), ("pain_canonical", 2), ("customer", 6)]:
        data[kind].x = torch.arange(count * 4, dtype=torch.float32).reshape(count, 4) / 10
    for relation, pairs in [
        (("vendor", "offers", "product"), [[0, 0], [1, 1]]),
        (("product", "provides", "cap_canonical"), [[0, 0], [1, 1]]),
        (("cap_canonical", "addresses", "pain_canonical"), [[0, 0], [1, 1]]),
        (("customer", "experiences", "pain_canonical"), [[0, 0], [1, 1]]),
    ]:
        data[relation].edge_index = torch.tensor(pairs).t().contiguous()
    return data


class ReconstructionTests(unittest.TestCase):
    def test_degree_is_unique_products_not_edge_count_or_per_vendor_top_100(self):
        base = base_fixture()
        base["product"].x = torch.zeros((101, 4))
        base["vendor", "offers", "product"].edge_index = torch.stack([
            torch.zeros(101, dtype=torch.long), torch.arange(101),
        ])
        base["product", "provides", "cap_canonical"].edge_index = torch.tensor([
            list(range(101)) + [0] * 200, [0] * 101 + [1] * 200,
        ])
        graph, _, report = collapsed_graph(base, [(0, 0)])
        self.assertEqual(report["capabilities_after"], 1)
        self.assertEqual(graph["vendor_product", "provides", "cap_canonical"].edge_index.shape[1], 200)

    def test_one_epoch_cpu_optimizer_and_validation(self):
        torch.set_num_threads(2)
        config = Configuration(hidden_dim=4, num_layers=1, dropout=0, epochs=1, patience=1)
        with patch("paper_submission.rgcn.info_nce", wraps=info_nce) as objective:
            state, vendor, customer, report = train(
                base_fixture(), [(0, 0), (1, 1)], [(0, 2), (1, 3)],
                config=config, seed=42,
            )
        self.assertIn(2, objective.call_args_list[0].args[3][0])
        self.assertNotIn(2, objective.call_args_list[1].args[3][0])
        self.assertTrue(state)
        self.assertEqual(tuple(vendor.shape), (2, 4))
        self.assertEqual(tuple(customer.shape), (6, 4))
        self.assertTrue(torch.isfinite(vendor).all())
        self.assertEqual(report["best_epoch"], 1)

    def test_negative_mask_does_not_consult_final_test(self):
        candidates = negative_candidates([(0, 0), (0, 1)], 4)
        self.assertEqual(candidates[0], [2, 3])

    def test_benchmark_enforces_actual_required_population(self):
        vendor = torch.ones((1, 4))
        customer = torch.ones((4, 4))
        with self.assertRaisesRegex(ValueError, "50 eligible sellers"):
            sampled_recovery(vendor, customer, [], [(0, 0)], seed=42)
        report = sampled_recovery(
            vendor, customer, [], [(0, 0)], seed=42,
            resamples=10, sellers_per_sample=1, positives=1, negatives=2, ks=(1, 2),
        )
        self.assertEqual(report["random_expected_precision"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
