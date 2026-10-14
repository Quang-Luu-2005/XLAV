"""Check the standalone notebook without model weights, network or a GPU."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image, ImageDraw


NOTEBOOK = Path(__file__).resolve().parents[1] / "source" / "e1_train_eval_phrasecut.ipynb"


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            tree = ast.parse(source)
            nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in {"rasterize_gt", "binary_metrics", "summarize"})
    namespace = {"np": np, "Image": Image, "ImageDraw": ImageDraw}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class BaselineNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = notebook_functions()

    def test_clean_notebook_and_all_cells_parse(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len({cell["id"] for cell in notebook["cells"]}), len(notebook["cells"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                ast.parse("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])

    def test_polygon_union_xy_fractional_and_clipping(self):
        # First rectangle rounds to x=0..1, y=1..2; second clips at right edge.
        task = {"task_id": "sample", "Polygons": [
            [[[0.8, 1.9], [1.8, 1.9], [1.8, 2.9], [0.8, 2.9]]],
            [[[3, 0], [7, 0], [7, 1], [3, 1]]],
        ]}
        expected = np.zeros((3, 4), dtype=bool)
        expected[1:3, 0:2] = True
        expected[0:2, 3] = True
        np.testing.assert_array_equal(self.functions["rasterize_gt"](task, 4, 3), expected)

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

    def test_empty_or_misaligned_ground_truth_rejected(self):
        with self.assertRaises(ValueError):
            self.functions["rasterize_gt"]({"task_id": "empty", "Polygons": []}, 4, 3)
        with self.assertRaises(ValueError):
            self.functions["binary_metrics"](np.ones((2, 2)), np.ones((3, 2), dtype=bool), 0.3)
        with self.assertRaises(ValueError):
            self.functions["binary_metrics"](np.ones((2, 2)), np.zeros((2, 2), dtype=bool), 0.3)


if __name__ == "__main__":
    unittest.main()
