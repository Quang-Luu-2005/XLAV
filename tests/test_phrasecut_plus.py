"""Check the PhraseCut+ helpers of the E1 notebook against the original CLIPSeg behaviour."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F


NOTEBOOK = Path(__file__).resolve().parents[1] / "source" / "e1_train_eval_phrasecut.ipynb"
NAMES = {"random_square_crop", "gaussian_blur", "visual_prompt", "mix_conditionals"}


def notebook_functions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    nodes = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse(cell["source"])
            nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES)
    namespace = {"torch": torch, "F": F, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class PhraseCutPlusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = notebook_functions()
        missing = NAMES - set(cls.f)
        assert not missing, missing

    def test_square_crop_keeps_object(self):
        torch.manual_seed(0)
        target = np.zeros((40, 100), dtype=bool)
        target[10:30, 80:95] = True  # object near the right edge
        y, x, side = self.f["random_square_crop"](target)
        self.assertEqual((y, side), (0, 40))
        self.assertGreater(target[y:y + side, x:x + side].sum(), 40 * 100 * 0.05)

    def test_square_crop_falls_back_to_best_crop(self):
        torch.manual_seed(0)
        target = np.zeros((10, 50), dtype=bool)
        target[0, 49] = True  # never above 5% of the image
        y, x, side = self.f["random_square_crop"](target, iterations=200)
        self.assertEqual(side, 10)
        self.assertTrue(target[y:y + side, x:x + side].any())

    def test_blur_preserves_constant_and_mean(self):
        image = torch.full((3, 32, 32), 0.7)
        torch.testing.assert_close(self.f["gaussian_blur"](image), image)
        noise = torch.rand(3, 40, 40, generator=torch.Generator().manual_seed(1))
        blurred = self.f["gaussian_blur"](noise)
        self.assertEqual(blurred.shape, noise.shape)
        self.assertLess(blurred.std(), noise.std())

    def test_visual_prompt_crops_object_and_dims_background(self):
        image = torch.ones(3, 100, 100)
        mask = torch.zeros(100, 100)
        mask[40:60, 20:80] = 1  # 20 x 60 object
        out = self.f["visual_prompt"](image, mask, size=64)
        self.assertEqual(out.shape, (3, 64, 64))
        # Crop 24 x 72 (10% context) padded to 72 x 72: zero padding top/bottom, object rows at 1.0.
        self.assertEqual(float(out[:, 0, :].abs().max()), 0.0)
        self.assertAlmostEqual(float(out[:, 32, 32].mean()), 1.0, places=5)
        # Context rows around the object keep 10% of the blurred background.
        row = round((24 + 1) * 64 / 72)
        self.assertLess(float(out[0, row, 32]), 0.5)

    def test_mix_uses_text_only_without_support(self):
        torch.manual_seed(0)
        text = torch.ones(4, 8)
        visual = torch.zeros(4, 8)
        valid = torch.tensor([True, True, False, False])
        mixed = self.f["mix_conditionals"](text, visual, valid, 0.5)
        torch.testing.assert_close(mixed[2:], text[2:])
        weights = mixed[:2, 0]
        self.assertTrue(((weights >= 0) & (weights <= 0.5)).all())


if __name__ == "__main__":
    unittest.main()
