"""Check paper-protocol result verification without model inference."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("paper_comparison",
    Path(__file__).resolve().parents[1] / "scripts/collect_e1_paper_comparison.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PaperComparisonTests(unittest.TestCase):
    def test_recomputed_iou_uses_all_queries_and_author_smoothing(self):
        stats = np.zeros((14354, 4, 49), dtype=np.int64)
        stats[:, 0] = 100
        stats[:, 1] = 25
        stats[:, 2] = 50
        stats[:, 3] = 352 * 352 - 175
        result = MODULE.recompute(stats, np.linspace(0, 1, 51)[1:-1])
        self.assertAlmostEqual(result["miou_0.3"], 100 / 176, places=12)
        self.assertAlmostEqual(result["fgiou_0.3"], 14354 * 100 / (1 + 14354 * 175), places=12)

    def test_partial_results_are_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.recompute(np.zeros((821, 4, 49), dtype=np.int64), np.linspace(0, 1, 51)[1:-1])


if __name__ == "__main__":
    unittest.main()
