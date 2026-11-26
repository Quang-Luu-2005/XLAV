"""E5 CLIPSeg + SAM notebook: prompt derivation, coordinates, fusion and selection, without SAM weights."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from transformers import SamImageProcessor, SamProcessor

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "source" / "e5_clipseg_sam.ipynb"
NAMES = {"prompt_components", "sam_coordinate_scale", "to_paper_size", "variant_probabilities", "query_iou",
         "paired_bootstrap"}


def cells():
    return {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}


def namespace():
    tree = ast.parse(cells()["e5-sam"])
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    assert {n.name for n in body} == NAMES
    ns = {"np": np, "torch": torch, "F": F, "ndimage": ndimage, "THRESHOLD": 0.3, "SAM_MAX_COMPONENTS": 4,
          "SAM_MIN_COMPONENT_RATIO": 0.1, "PAPER_SIZE": 352, "SEED": 42,
          "PAPER_T03": int(np.argmin(np.abs(np.linspace(0, 1, 51)[1:-1] - 0.3)))}
    exec(compile(ast.Module(body=body, type_ignores=[]), "e5-sam", "exec"), ns)
    return ns


class SamNotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def test_prompts_come_from_prediction_components(self):
        probability = np.zeros((40, 60), dtype=np.float32)
        probability[5:15, 10:30] = 0.6     # large component, box x 10..30, y 5..15
        probability[8, 20] = 0.9           # its most confident pixel
        probability[30:35, 50:55] = 0.5    # second component, 25 px >= 0.1 x 200
        probability[0, 0] = 0.4            # 1 px component, < 0.1 x 200: dropped
        prompts = self.ns["prompt_components"](probability)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][0], [10.0, 5.0, 30.0, 15.0])
        self.assertEqual(prompts[0][1], [20.5, 8.5])
        self.assertEqual(prompts[1][0], [50.0, 30.0, 55.0, 35.0])
        self.assertEqual(self.ns["prompt_components"](np.full((5, 5), 0.29, dtype=np.float32)), [])

    def test_component_limit(self):
        probability = np.zeros((10, 50), dtype=np.float32)
        for k in range(6):
            probability[2:5, k * 8:k * 8 + 3] = 0.8
        self.assertEqual(len(self.ns["prompt_components"](probability)), 4)

    def test_coordinate_scale_matches_sam_processor(self):
        processor = SamProcessor(image_processor=SamImageProcessor())
        for height, width in [(600, 800), (375, 500), (1024, 683), (333, 1000)]:
            sx, sy = self.ns["sam_coordinate_scale"](height, width)
            box = np.array([[12.0, 34.0, 456.0, 300.0]])
            expected = processor._normalize_coordinates(1024, box, (height, width), is_bounding_box=True)
            np.testing.assert_allclose(box * [sx, sy, sx, sy], expected)

    def test_paper_size_and_fusion(self):
        logits = torch.zeros(20, 30)
        logits[:, 15:] = 10.0
        probability = self.ns["to_paper_size"](logits)
        self.assertEqual(probability.shape, (352, 352))
        self.assertAlmostEqual(float(probability[0, 0]), 0.5, places=5)
        self.assertGreater(float(probability[0, -1]), 0.99)
        clip = np.full((352, 352), 0.2, dtype=np.float32)
        sam = np.full((352, 352), 0.8, dtype=np.float32)
        variants = self.ns["variant_probabilities"](clip, sam, sam)
        self.assertEqual(list(variants), ["clipseg", "sam_box_replace", "sam_box_mean",
                                          "sam_boxpoint_replace", "sam_boxpoint_mean"])
        np.testing.assert_allclose(variants["sam_box_mean"], 0.5)
        fallback = self.ns["variant_probabilities"](clip, None, None)
        for value in fallback.values():
            np.testing.assert_array_equal(value, clip)

    def test_paired_bootstrap(self):
        base = {"a": 0.5, "b": 0.5, "c": 0.4}
        other = {"a": 0.7, "b": 0.5, "c": 0.3}
        result = self.ns["paired_bootstrap"](base, other, {"a": 1, "b": 1, "c": 2}, reps=500)
        self.assertAlmostEqual(result["delta_miou_0.3"], (0.2 + 0.0 - 0.1) / 3)
        self.assertEqual((result["wins"], result["losses"], result["n_queries"]), (1, 1, 3))
        self.assertLessEqual(result["ci95"][0], result["delta_miou_0.3"])

    def test_lora_checkpoint_loads_and_hf_forward_matches_segment(self):
        import hashlib
        import tempfile
        from transformers import CLIPSegConfig, CLIPSegForImageSegmentation
        source = cells()
        helpers = {"torch": torch, "LORA_GRAD_CHECKPOINT": False, "LORA_LR_SCALE": 1.0}
        exec(compile(source["e5-lora-helpers"], "e5-lora-helpers", "exec"), helpers)

        def tiny():
            torch.manual_seed(0)
            return CLIPSegForImageSegmentation(CLIPSegConfig(
                text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                             "num_attention_heads": 2, "vocab_size": 99, "max_position_embeddings": 16},
                vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 5,
                               "num_attention_heads": 2, "image_size": 32, "patch_size": 16},
                projection_dim=16, reduce_dim=8, decoder_num_attention_heads=2, decoder_intermediate_size=16,
                extract_layers=[1, 2, 3], conditional_layer=0)).eval()
        trained = tiny()
        helpers["attach_lora"](trained, 2, 4)
        with torch.no_grad():
            for module in helpers["lora_modules"](trained):
                module.lora_b.normal_(std=0.5)
        pixels, ids = torch.randn(2, 3, 48, 48), torch.tensor([[0, 5, 7, 1], [0, 9, 1, 1]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            expected = trained(pixel_values=pixels, input_ids=ids, attention_mask=mask).logits
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder_epoch_002.pt"
            torch.save({"decoder": trained.decoder.state_dict(), "lora": helpers["lora_state_dict"](trained),
                        "epoch": 2, "global_step": 9709, "base_model_id": "tiny", "base_model_revision": "pinned",
                        "training": {"arm": "E4c", "lora": {"rank": 2, "alpha": 4}}}, path)
            ns = {**helpers, "model": tiny(), "TRAIN_DECODER": False, "DECODER_CHECKPOINT": str(path), "Path": Path,
                  "MODEL_ID": "tiny", "MODEL_REVISION": "pinned", "LORA_RANK": 8, "LORA_ALPHA": 16,
                  "sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), "print": lambda *a, **k: None}
            exec(compile(source["cell-15"], "cell-15", "exec"), ns)
        loaded = ns["model"]
        self.assertTrue(helpers["has_lora"](loaded))
        with torch.no_grad():
            hf = loaded(pixel_values=pixels, input_ids=ids, attention_mask=mask).logits
            torch.testing.assert_close(hf, expected)
            torch.testing.assert_close(hf, helpers["segment"](loaded, pixels, ids, mask))
        self.assertIn("forward khớp segment", source["e5-lora-check"])

    def test_notebook_structure(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        order = [c["id"] for c in notebook["cells"]]
        source = cells()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
        config = source["cell-04"]
        for line in ['EVAL_SPLIT = "val"', "TRAIN_DECODER = False", "EVAL_AMP = False", "MAX_QUERIES = None",
                     'SAM_MODEL = "facebook/sam-vit-base"']:
            self.assertIn(line, config)
        # Selection is written before test is evaluated; ground truth only enters after prompts are built.
        self.assertLess(order.index("e5-run-val"), order.index("e5-switch-test"))
        self.assertLess(order.index("e5-switch-test"), order.index("e5-run-test"))
        self.assertIn('write_json(OUT / "selection.json"', source["e5-run-val"])
        run = source["e5-sam"][source["e5-sam"].index("def run_split"):]
        self.assertLess(run.index("prompt_components(probability)"), run.index("rasterize_gt("))
        self.assertIn('OUT = WORK_ROOT / "clipseg_e5_sam"', source["cell-05"])
        self.assertIn("clipseg_test_reproduces_e1", source["e5-run-test"])


if __name__ == "__main__":
    unittest.main()
