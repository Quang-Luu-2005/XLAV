"""Check the E2/E3 notebooks without model weights, network or a GPU."""
import ast
import json
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "source"
NOTEBOOKS = {
    "e2": SOURCE / "e2_text_vs_visual_phrasecut_miniv.ipynb",
    "e3": SOURCE / "e3_prompt_templates_phrasecut_miniv.ipynb",
}


def notebook_functions(path, names):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse(cell["source"])
            nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    namespace = {"np": np, "Image": Image, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ExperimentNotebookTests(unittest.TestCase):
    def test_clean_notebooks_and_all_cells_parse(self):
        for path in NOTEBOOKS.values():
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            self.assertEqual(len({cell["id"] for cell in notebook["cells"]}), len(notebook["cells"]))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    ast.parse(cell["source"])

    def test_paired_comparison_counts_and_ci(self):
        compare = notebook_functions(NOTEBOOKS["e3"], {"paired_comparison"})["paired_comparison"]
        base = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        other = {"a": 0.7, "b": 0.6, "c": 0.5, "d": 0.495}
        image_of = {"a": 1, "b": 1, "c": 2, "d": 3}
        result = compare(base, other, image_of, 500, 0.01, 42)
        self.assertAlmostEqual(result["delta_mean_iou"], (0.2 + 0.1 + 0 - 0.005) / 4)
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (2, 2, 0))
        self.assertLessEqual(result["ci95_low"], result["delta_mean_iou"])
        self.assertGreaterEqual(result["ci95_high"], result["delta_mean_iou"])
        with self.assertRaises(ValueError):
            compare(base, {"a": 0.1}, image_of, 100, 0.01, 42)

    def test_support_views_use_support_mask_only(self):
        views = notebook_functions(NOTEBOOKS["e2"], {"support_views"})["support_views"]
        image = Image.fromarray(np.full((20, 30, 3), 200, dtype=np.uint8))
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 10:20] = True
        result = views(image, mask, 0.1, 0.1)
        self.assertEqual(result["visual_crop"].size, (12, 12))  # 10 px box + 1 px padding each side
        dark = np.asarray(result["visual_dark"])
        self.assertEqual(dark[10, 15, 0], 200)
        self.assertEqual(dark[0, 0, 0], 20)
        self.assertEqual(result["visual_crop_dark"].size, (12, 12))
        self.assertEqual(np.asarray(result["visual_crop_dark"])[0, 0, 0], 20)


if __name__ == "__main__":
    unittest.main()
