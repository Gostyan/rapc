from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.aggregate import aggregate
from experiments.evaluate import evaluate
from experiments.select_loco import select
from experiments.select_loco_multiseed import select_multiseed


def write_cache(path: Path, embeddings, labels, domains, unseen_mask=None) -> None:
    arrays = {
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "domains": np.asarray(domains, dtype=np.int64),
    }
    if unseen_mask is not None:
        arrays["unseen_mask"] = np.asarray(unseen_mask, dtype=bool)
    np.savez(path, **arrays)


class StandaloneExperimentTests(unittest.TestCase):
    def test_evaluation_infers_missing_training_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.npz"
            evaluation = root / "evaluation.npz"
            lock = Path(__file__).resolve().parents[1] / "configs/final_rapc.json"
            write_cache(
                training,
                [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.7, 0.7]],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            )
            write_cache(evaluation, [[0, 1], [0.1, 0.9]], [1, 1], [1, 1])
            result = evaluate(training, evaluation, lock)
            self.assertEqual(result["methods"]["rapc"]["num_samples"], 2)
            self.assertFalse(result["protocol"]["query_labels_used_for_prototypes"])

    def test_registered_mask_may_select_a_subset_of_missing_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.npz"
            evaluation = root / "evaluation.npz"
            lock = Path(__file__).resolve().parents[1] / "configs/final_rapc.json"
            write_cache(training, [[1, 0], [0, 1], [0.7, 0.7]], [0, 1, 0], [0, 0, 1])
            write_cache(
                evaluation, [[0, 1], [1, 0]], [1, 0], [1, 0],
                unseen_mask=[True, False],
            )
            result = evaluate(training, evaluation, lock)
            self.assertEqual(result["methods"]["rapc"]["num_samples"], 1)
            self.assertIn("local_rapc", result["methods"])

    def test_loco_selection_has_no_test_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.npz"
            validation = root / "validation.npz"
            embeddings = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]
            labels = [0, 0, 1, 1, 0, 1]
            domains = [0, 0, 0, 0, 1, 1]
            write_cache(training, embeddings, labels, domains)
            write_cache(validation, embeddings, labels, domains)
            result = select(training, validation, lambdas=(1.0,), strengths=(0.0, 0.75))
            self.assertFalse(result["protocol"]["test_cache_read"])
            self.assertGreater(result["selected"]["cell_macro_recall"], -1.0)

    def test_multiseed_loco_pools_cell_seed_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings = [[1, 0], [0, 1], [0.8, 0.2], [0.2, 0.8]]
            labels = [0, 1, 0, 1]
            domains = [0, 0, 1, 1]
            training_paths = []
            validation_paths = []
            for seed in (42, 43):
                training = root / f"seed{seed}_training.npz"
                validation = root / f"seed{seed}_validation.npz"
                write_cache(training, embeddings, labels, domains)
                write_cache(validation, embeddings, labels, domains)
                training_paths.append(training)
                validation_paths.append(validation)
            result = select_multiseed(
                training_paths,
                validation_paths,
                lambdas=(1.0,),
                strengths=(0.75,),
            )
            self.assertEqual(len(result["runs"]), 2)
            self.assertFalse(result["protocol"]["test_cache_read"])

    def test_matched_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, gain in enumerate((0.01, 0.02, -0.01)):
                path = Path(directory) / f"seed{index}.json"
                payload = {
                    "status": "complete",
                    "methods": {
                        "global_prototype": {"BA_unseen": 0.3},
                        "rapc": {"BA_unseen": 0.3 + gain},
                    },
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            result = aggregate(paths)
            self.assertEqual(result["run_count"], 3)
            self.assertEqual(result["positive_runs"], 2)


if __name__ == "__main__":
    unittest.main()
