"""Check the standalone notebook without model weights, network or a GPU."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import simpson
from skimage.draw import polygon2mask


NOTEBOOK = Path(__file__).resolve().parents[1] / "source" / "e1_train_eval_phrasecut.ipynb"
NAMES = {"rasterize_gt", "binary_metrics", "summarize", "paper_target", "paper_threshold_stats", "paper_metrics"}


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            tree = ast.parse(source)
            nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES)
    namespace = {"np": np, "torch": torch, "F": F, "polygon2mask": polygon2mask, "simpson": simpson,
                 "PAPER_SIZE": 352, "PAPER_THRESHOLDS": np.linspace(0, 1, 51)[1:-1]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def author_fixed_interval_metrics(predictions, targets):
    """Port of FixedIntervalMetrics.add/value (timojl/clipseg metrics.py), each query its own class."""
    thresholds = np.linspace(0, 1, 51)[1:-1]
    stats = {"tp": [], "fp": [], "fn": [], "tn": []}
    for prediction, target in zip(predictions, targets):
        p, g = torch.tensor(prediction).flatten(), torch.tensor(target).flatten()
        p_sorted = p.sort()
        p, g = p_sorted.values, g[p_sorted.indices]
        tps, fps, fns, tns = [], [], [], []
        for thresh in thresholds:
            valid = torch.where(p > thresh)[0]
            n = int(valid[0]) if len(valid) > 0 else len(g)
            fn, tp = int(g[:n].sum()), int(g[n:].sum())
            fns.append(fn); tns.append(n - fn); tps.append(tp); fps.append(len(g) - n - tp)
        for key, value in zip(["tp", "fp", "fn", "tn"], [tps, fps, fns, tns]):
            stats[key].append(value)
    summed = {k: np.array(v).sum(0) for k, v in stats.items()}
    keep = range(len(thresholds))
    precisions = [summed["tp"][j] / (1 + summed["tp"][j] + summed["fp"][j]) for j in keep if summed["tp"][j] + summed["fp"][j] > 0]
    recalls = [summed["tp"][j] / (1 + summed["tp"][j] + summed["fn"][j]) for j in keep if summed["tp"][j] + summed["fp"][j] > 0]
    recalls, precisions = zip(*sorted(list(set(zip(recalls, precisions))), key=lambda x: x[0]))
    fgiou = [summed["tp"][j] / (1 + summed["tp"][j] + summed["fp"][j] + summed["fn"][j]) for j in keep]
    per_query = {k: np.array(v) for k, v in stats.items()}
    miou = (per_query["tp"] / (1 + per_query["tp"] + per_query["fp"] + per_query["fn"])).mean(0)
    i3, i5 = list(np.round(thresholds, 6)).index(0.3), list(np.round(thresholds, 6)).index(0.5)
    return {"ap": simpson(precisions, x=recalls), "fgiou_0.3": fgiou[i3], "fgiou_0.5": fgiou[i5], "miou_0.3": miou[i3]}, per_query


class BaselineNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = notebook_functions()
        assert not NAMES - set(cls.functions), NAMES - set(cls.functions)

    def test_clean_notebook_and_all_cells_parse(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len({cell["id"] for cell in notebook["cells"]}), len(notebook["cells"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])

    def test_polygon_union_matches_skimage_with_xy_swapped(self):
        # Polygons are (x, y); the image is 5 rows x 8 columns, so a transposed mask would fail.
        task = {"task_id": "sample", "Polygons": [
            [[[5, 1], [7, 1], [7, 3], [5, 3]]],
            [[[0, 0], [2, 0], [2, 1]], [[1, 1], [3, 1]]],  # second polygon has < 3 points: ignored
        ]}
        expected = (polygon2mask((5, 8), np.array([[1, 5], [1, 7], [3, 7], [3, 5]], float))
                    | polygon2mask((5, 8), np.array([[0, 0], [0, 2], [1, 2]], float)))
        result = self.functions["rasterize_gt"](task, 8, 5)
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(result[2, 6])
        self.assertFalse(result[3, 1])  # ngoài cả hai polygon

    def test_empty_ground_truth_rejected_unless_allowed(self):
        with self.assertRaises(ValueError):
            self.functions["rasterize_gt"]({"task_id": "empty", "Polygons": []}, 4, 3)
        empty = self.functions["rasterize_gt"]({"task_id": "empty", "Polygons": []}, 4, 3, allow_empty=True)
        self.assertFalse(empty.any())

    def test_paper_target_nearest_resize(self):
        target = np.array([[1, 0], [0, 0]], dtype=bool)
        resized = self.functions["paper_target"](target)
        self.assertEqual(resized.shape, (352, 352))
        self.assertTrue(resized[:176, :176].all())
        self.assertFalse(resized[176:, :].any() or resized[:, 176:].any())

    def test_paper_metrics_match_author_fixed_interval_metrics(self):
        rng = np.random.default_rng(0)
        predictions = [rng.random((24, 24)).astype(np.float32) for _ in range(6)]
        targets = [rng.random((24, 24)) < q for q in [0.05, 0.2, 0.5, 0.01, 0.0, 0.3]]
        targets[1] |= predictions[1] > 0.7  # some correlated queries so the PR curve is non-trivial
        expected, per_query = author_fixed_interval_metrics(predictions, targets)
        stats = np.stack([self.functions["paper_threshold_stats"](p, g) for p, g in zip(predictions, targets)])
        for i, key in enumerate(["tp", "fp", "fn", "tn"]):
            np.testing.assert_array_equal(stats[:, i], per_query[key])
        result = self.functions["paper_metrics"](stats)
        for key in ["ap", "fgiou_0.3", "fgiou_0.5", "miou_0.3"]:
            self.assertAlmostEqual(result[key], float(expected[key]), places=12, msg=key)

    def test_binary_overlap_and_threshold_boundary(self):
        row = self.functions["binary_metrics"](
            np.array([[0.3, 0.8], [0.2, 0.1]]), np.array([[1, 0], [1, 0]], dtype=bool), 0.3)
        self.assertEqual(row["intersection"], 1)
        self.assertEqual(row["union"], 3)
        self.assertAlmostEqual(row["iou"], 1 / 3)
        self.assertAlmostEqual(row["dice"], 0.5)

    def test_sample_mean_differs_from_cumulative_iou(self):
        rows = [dict(image_id=1, iou=0.5, dice=2/3, intersection=1, union=2),
                dict(image_id=2, iou=0.8, dice=8/9, intersection=8, union=10)]
        result = self.functions["summarize"](rows)
        self.assertAlmostEqual(result["mean_sample_iou"], 0.65)
        self.assertAlmostEqual(result["cumulative_iou"], 0.75)
        self.assertEqual(result["n_queries"], 2)
        self.assertEqual(result["n_images"], 2)

    def test_misaligned_ground_truth_rejected(self):
        with self.assertRaises(ValueError):
            self.functions["binary_metrics"](np.ones((2, 2)), np.ones((3, 2), dtype=bool), 0.3)
        with self.assertRaises(ValueError):
            self.functions["binary_metrics"](np.ones((2, 2)), np.zeros((2, 2), dtype=bool), 0.3)


if __name__ == "__main__":
    unittest.main()
