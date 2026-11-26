"""Check the E2/E3 notebooks without model weights, network or a GPU."""
import ast
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "source"
NOTEBOOKS = {
    "e2": SOURCE / "e2_text_vs_visual.ipynb",
    "e3": SOURCE / "e3_prompt_templates.ipynb",
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
    def test_training_matched_support_is_exact_training_transform(self):
        functions = notebook_functions(NOTEBOOKS["e2"], {"training_support", "random_square_crop", "visual_prompt", "gaussian_blur"})
        mean, std = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
        functions.update(torch=torch, F=F, DARK_FACTOR=0.1, CROP_PADDING=0.1, SUPPORT_AUG_CROP=True, SEED=42,
                         image_mean=mean, image_std=std)
        rgb = np.random.default_rng(5).integers(0, 256, (83, 310, 3), dtype=np.uint8)
        mask = np.zeros((83, 310), dtype=bool)
        mask[20:40, 100:160] = True
        result = functions["training_support"](Image.fromarray(rgb), mask, seed=7)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            y, x, side = functions["random_square_crop"](mask)
        pixels = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[:, y:y+side, x:x+side]
        pixels = F.interpolate(pixels[None], size=(352,352), mode="bilinear", align_corners=True)[0]
        target = torch.from_numpy(mask[y:y+side, x:x+side].astype(np.float32))
        target = F.interpolate(target[None,None], size=(352,352), mode="nearest")[0,0]
        expected = functions["visual_prompt"]((pixels/255-mean)/std, target)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (3, 352, 352))

    def test_e2_all_eligible_and_capped_selection(self):
        notebook = json.loads(NOTEBOOKS["e2"].read_text(encoding="utf-8"))
        source = next(c["source"] for c in notebook["cells"] if c["id"] == "cell-07")
        source = source[source.index("assert SUPPORTS_PER_QUERY"):]
        tasks = [{"task_id": str(i), "image_id": i, "phrase": "person",
                  "phrase_structure": {"name": "person"}, "instance_boxes": [[0, 0, 5, 5]]}
                 for i in range(1, 7)]
        tasks.append({**tasks[0], "task_id": "rare", "phrase_structure": {"name": "rare"}})
        for limit in (None, 3):
            namespace = {"tasks": tasks, "info_by_id": {i: {"width": 10, "height": 10} for i in range(1, 7)},
                         "defaultdict": defaultdict, "random": random, "SUPPORTS_PER_QUERY": 3,
                         "MAX_QUERIES": limit, "MIN_SUPPORT_BOX_FRACTION": 0.02, "SEED": 42,
                         "CALIBRATION_FRACTION": 0.2, "OUT": Path("unused"),
                         "write_json": lambda *args: None, "print": lambda *args: None}
            exec(compile(source, "E2 selection", "exec"), namespace)
            pairs = namespace["pairs"]
            self.assertEqual(len(pairs), 6 if limit is None else 3)
            self.assertNotIn("rare", {p["query"]["task_id"] for p in pairs})
            for pair in pairs:
                supports = {s["image_id"] for s in pair["supports"]}
                self.assertEqual(len(supports), 3)
                self.assertNotIn(pair["query"]["image_id"], supports)

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
