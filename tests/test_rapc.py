from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from rapc import (
    build_cell_table,
    build_domain_prototypes,
    fit_rapc,
    fit_two_way_ridge,
    predict,
    support_connected,
)


class RAPCTests(unittest.TestCase):
    def test_closed_form_recovers_exact_additive_cell_without_ridge(self) -> None:
        class_effect = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        domain_effect = torch.tensor([[0.0, 0.0], [3.0, -1.0]])
        pairs = [(0, 0), (0, 1), (1, 0)]
        centers = torch.stack([class_effect[c] + domain_effect[d] for d, c in pairs])
        coefficients, metadata = fit_two_way_ridge(
            centers,
            torch.tensor([c for _, c in pairs]),
            torch.tensor([d for d, _ in pairs]),
            num_classes=2,
            num_domains=2,
            ridge_lambda=0.0,
        )
        query = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertTrue(torch.allclose(query @ coefficients, torch.tensor([3.0, 1.0]), atol=1e-6))
        self.assertTrue(metadata["closed_form"])
        self.assertFalse(metadata["unique_solution"])

    def test_positive_ridge_has_unique_solution(self) -> None:
        centers = torch.tensor([[1.0], [2.0], [3.0]])
        _, metadata = fit_two_way_ridge(
            centers,
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            num_classes=2,
            num_domains=2,
            ridge_lambda=1.0,
        )
        self.assertTrue(metadata["global_optimum"])
        self.assertTrue(metadata["unique_solution"])

    def test_disconnected_support_is_rejected(self) -> None:
        observed = torch.tensor([[True, False], [False, True]])
        self.assertFalse(support_connected(observed, target_domain=1, target_class=0))
        self.assertTrue(support_connected(observed, target_domain=1, target_class=1))

    def test_only_missing_entries_are_completed(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]]
        )
        classes = torch.tensor([0, 0, 1, 1, 0])
        domains = torch.tensor([0, 0, 0, 0, 1])
        fitted = fit_rapc(
            embeddings, classes, domains, num_classes=2, num_domains=2
        )
        prototypes, details = build_domain_prototypes(fitted, target_domain=1)
        expected_global = F.normalize(fitted.table.global_centers, dim=1)
        self.assertTrue(torch.allclose(prototypes[0], expected_global[0]))
        self.assertEqual(details[0]["role"], "observed_global_anchor")
        self.assertEqual(details[1]["role"], "missing_completed")
        self.assertAlmostEqual(float(prototypes[1].norm()), 1.0, places=6)

    def test_local_policy_changes_only_observed_entries(self) -> None:
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
        fitted = fit_rapc(
            embeddings, torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1]),
            num_classes=2, num_domains=2,
        )
        global_table, _ = build_domain_prototypes(fitted, 1)
        local_table, details = build_domain_prototypes(
            fitted, 1, observed_policy="local"
        )
        self.assertFalse(torch.equal(global_table[0], local_table[0]))
        self.assertTrue(torch.equal(global_table[1], local_table[1]))
        self.assertEqual(details[0]["role"], "observed_local_cell")

    def test_prediction_requires_domains_but_not_query_labels(self) -> None:
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.7, 0.7]]
        )
        fitted = fit_rapc(
            embeddings,
            torch.tensor([0, 0, 1, 1, 0]),
            torch.tensor([0, 0, 0, 0, 1]),
            num_classes=2,
            num_domains=2,
        )
        predictions = predict(
            fitted,
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([0, 1]),
        )
        self.assertEqual(tuple(predictions.shape), (2,))

    def test_cell_centers_are_equal_cell_statistics(self) -> None:
        table = build_cell_table(
            torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 0, 1]),
            num_classes=2,
            num_domains=2,
        )
        self.assertEqual(table.counts.tolist(), [[2, 0], [0, 1]])
        self.assertTrue(torch.allclose(table.centers[0, 0], torch.tensor([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
