import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from clipseg_project.metrics import measure, summarize


class MetricTests(unittest.TestCase):
    def test_hand_computed_overlap(self):
        row = measure(np.array([[0.9, 0.8], [0.2, 0.1]]), [[1, 0], [1, 0]], 0.3)
        self.assertAlmostEqual(row["iou"], 1 / 3)
        self.assertAlmostEqual(row["dice"], 0.5)
        self.assertAlmostEqual(row["pixel_ap"], 5 / 6)

    def test_cumulative_is_not_mean_sample_iou(self):
        rows = [measure(np.ones((1, 1)), np.ones((1, 1))),
                measure(np.zeros((3, 3)), np.ones((3, 3)))]
        result = summarize(rows)
        self.assertAlmostEqual(result["mean_sample_iou"], 0.5)
        self.assertAlmostEqual(result["cumulative_iou"], 0.1)

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            measure([[1.2]], [[1]])
        with self.assertRaises(ValueError):
            measure([[0.5]], [[0]])
        with self.assertRaises(ValueError):
            measure([[0.5]], [[1, 0]])


if __name__ == "__main__":
    unittest.main()
