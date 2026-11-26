"""E4c LoRA notebook, checked on a tiny random CLIPSeg without network or GPU."""
import ast
import hashlib
import math
import json
import tempfile
from pathlib import Path
import unittest

import torch
from transformers import CLIPSegConfig, CLIPSegForImageSegmentation

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "source" / "e4_lora_clip.ipynb"


def module_namespace():
    """Classes and functions of the e4-module cell, without its forward check against a loaded model."""
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
    tree = ast.parse(cells["e4-module"])
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    namespace = {"torch": torch, "LORA_GRAD_CHECKPOINT": False}
    exec(compile(ast.Module(body=body, type_ignores=[]), "e4-module", "exec"), namespace)
    return namespace


def tiny_model():
    torch.manual_seed(0)
    config = CLIPSegConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "vocab_size": 99, "max_position_embeddings": 16},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 5, "num_attention_heads": 2,
                       "image_size": 32, "patch_size": 16},
        projection_dim=16, reduce_dim=8, decoder_num_attention_heads=2, decoder_intermediate_size=16,
        extract_layers=[1, 2, 3], conditional_layer=0, use_complex_transposed_convolution=True)
    return CLIPSegForImageSegmentation(config).eval()


class E4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.ns = module_namespace()
        self.model = tiny_model()
        torch.manual_seed(1)
        self.pixels = torch.randn(2, 3, 48, 48)
        self.ids = torch.tensor([[0, 5, 7, 1], [0, 9, 1, 1]])
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])

    def lora_model(self, std_b=0.0):
        if not self.ns["has_lora"](self.model):
            self.ns["attach_lora"](self.model, rank=2, alpha=4)
        with torch.no_grad():
            for module in self.ns["lora_modules"](self.model):
                module.lora_b.normal_(std=std_b) if std_b else module.lora_b.zero_()
        return self.model

    def test_segment_matches_transformers_forward(self):
        with torch.no_grad():
            reference = self.model(pixel_values=self.pixels, input_ids=self.ids, attention_mask=self.mask).logits
            result = self.ns["segment"](self.model, self.pixels, self.ids, self.mask)
        self.assertEqual(result.shape, (2, 48, 48))
        torch.testing.assert_close(result, reference)

    def test_vision_activations_match_transformers_hidden_states(self):
        with torch.no_grad():
            hidden = self.model.clip.vision_model(pixel_values=self.pixels, output_hidden_states=True).hidden_states
            ours = self.ns["vision_activations"](self.model, self.pixels)
        self.assertEqual(len(ours), 3)
        for a, b in zip(ours, [hidden[i + 1] for i in self.model.extract_layers]):
            torch.testing.assert_close(a, b)

    def test_lora_targets_only_layers_feeding_decoder(self):
        model = self.lora_model()
        modules = self.ns["lora_modules"](model)
        self.assertEqual(len(modules), 2 * (max(model.extract_layers) + 1))  # q/v of layers 0..3, not layer 4
        self.assertEqual(sum(p.numel() for p in self.ns["lora_parameters"](model)), len(modules) * 2 * 2 * 32)
        with self.assertRaises(AssertionError):
            self.ns["attach_lora"](model, rank=2, alpha=4)

    def test_zero_lora_reproduces_base_and_conditioning_stays_frozen(self):
        segment = self.ns["segment"]
        supports = torch.randn_like(self.pixels)
        weights = torch.full((2, 1), 0.3)
        with torch.no_grad():
            before = segment(self.model, self.pixels, self.ids, self.mask, supports, weights)
            visual_before = self.model.get_conditional_embeddings(batch_size=2, conditional_pixel_values=supports)
            torch.testing.assert_close(segment(self.lora_model(0.0), self.pixels, self.ids, self.mask, supports, weights),
                                       before)
            model = self.lora_model(0.5)
            changed = segment(model, self.pixels, self.ids, self.mask, supports, weights)
            with self.ns["lora_disabled"](model):
                visual_after = model.get_conditional_embeddings(batch_size=2, conditional_pixel_values=supports)
        self.assertGreater((changed - before).abs().max().item(), 1e-4)  # LoRA changes query features
        torch.testing.assert_close(visual_after, visual_before)  # support embedding uses frozen CLIP
        self.assertTrue(all(m.enabled for m in self.ns["lora_modules"](model)))

    def test_experiment_visual_path_keeps_frozen_support_embedding(self):
        model = self.lora_model(0.5)
        supports = torch.randn_like(self.pixels)
        with torch.no_grad():
            actual = self.ns["experiment_logits"](model, self.pixels, conditional_pixel_values=supports)
            with self.ns["lora_disabled"](model):
                cond = model.get_conditional_embeddings(batch_size=2, conditional_pixel_values=supports)
            expected = model.decoder(self.ns["vision_activations"](model, self.pixels), cond).logits
        torch.testing.assert_close(actual, expected)

    def test_lora_gradients_reach_adapters_not_clip_weights(self):
        model = self.lora_model(0.0)
        model.clip.requires_grad_(False)
        for parameter in self.ns["lora_parameters"](model):
            parameter.requires_grad_(True)
        self.ns["segment"](model, self.pixels, self.ids, self.mask).mean().backward()
        self.assertTrue(all(m.lora_b.grad is not None and m.lora_b.grad.abs().sum() > 0
                            for m in self.ns["lora_modules"](model)))
        self.assertTrue(all(p.grad is None for n, p in model.clip.named_parameters() if "lora_" not in n))

    def test_gradient_checkpointing_gives_same_gradients(self):
        grads = []
        for flag in [False, True]:
            self.ns["LORA_GRAD_CHECKPOINT"] = flag
            model = tiny_model()
            self.ns["attach_lora"](model, rank=2, alpha=4)
            with torch.no_grad():
                for module in self.ns["lora_modules"](model):
                    module.lora_b.fill_(0.1)
            model.clip.requires_grad_(False)
            for parameter in self.ns["lora_parameters"](model):
                parameter.requires_grad_(True)
            self.ns["segment"](model, self.pixels, self.ids, self.mask).mean().backward()
            grads.append([p.grad.clone() for p in self.ns["lora_parameters"](model)])
        self.ns["LORA_GRAD_CHECKPOINT"] = False
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b)

    def test_lora_state_round_trip(self):
        model = self.lora_model(0.5)
        state = self.ns["lora_state_dict"](model)
        self.assertTrue(state and all("lora_" in k for k in state))
        with torch.no_grad():
            expected = self.ns["segment"](model, self.pixels, self.ids, self.mask)
        fresh = tiny_model()
        self.ns["attach_lora"](fresh, rank=2, alpha=4)
        self.ns["load_lora_state"](fresh, state)
        with torch.no_grad():
            torch.testing.assert_close(self.ns["segment"](fresh, self.pixels, self.ids, self.mask), expected)

    def test_e2_loads_lora_checkpoint_and_preserves_logits(self):
        model = self.lora_model(0.5)
        with torch.no_grad():
            expected_text = self.ns["experiment_logits"](model, self.pixels, self.ids, self.mask)
            expected_visual = self.ns["experiment_logits"](model, self.pixels, conditional_pixel_values=self.pixels)
        builder = ast.parse((ROOT / "scripts/build_experiment_notebooks.py").read_text(encoding="utf-8"))
        source = next(ast.literal_eval(n.value) for n in builder.body if isinstance(n, ast.Assign)
                      and getattr(n.targets[0], "id", None) == "LOAD_DECODER")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.pt"
            torch.save({"decoder": model.decoder.state_dict(), "lora": self.ns["lora_state_dict"](model),
                        "epoch": 2, "global_step": 4, "base_model_id": "tiny", "base_model_revision": "pinned",
                        "training": {"arm": "E4c", "epochs": 2, "lora": {"rank": 2, "alpha": 4}}}, path)
            namespace = {**self.ns, "model": tiny_model(), "DECODER_CHECKPOINT": str(path), "Path": Path,
                "MODEL_ID": "tiny", "MODEL_REVISION": "pinned", "device": torch.device("cpu"), "json": json,
                "sha256": lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), "print": lambda *a, **kw: None}
            exec(source, namespace)
        loaded = namespace["model"]
        self.assertEqual(namespace["decoder_info"]["experiment_arm"], "E4c")
        self.assertEqual(namespace["decoder_info"]["lora"], {"rank": 2, "alpha": 4})
        with torch.no_grad():
            torch.testing.assert_close(self.ns["experiment_logits"](loaded, self.pixels, self.ids, self.mask),
                                       expected_text)
            torch.testing.assert_close(self.ns["experiment_logits"](loaded, self.pixels,
                                                                     conditional_pixel_values=self.pixels),
                                       expected_visual)

    def test_e2_e3_notebooks_embed_lora_helpers(self):
        for filename in ["e2_text_vs_visual.ipynb", "e3_prompt_templates.ipynb"]:
            source = "".join("".join(c["source"]) for c in json.loads((ROOT / "source" / filename).read_text(
                encoding="utf-8"))["cells"] if c["cell_type"] == "code")
            self.assertIn("class LoRALinear", source)
            self.assertIn("LORA_GRAD_CHECKPOINT = False", source)
            self.assertNotIn("WordAttention", source)

    def scratch_schedule(self, sessions, steps_per_epoch=4, lora_scale=None):
        """Run the notebook's epoch/stage logic with a toy optimizer; save/restore state between sessions."""
        import textwrap
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
        config = cells["cell-04"]
        stages = ast.literal_eval(config[config.index("SCRATCH_STAGES = ") + len("SCRATCH_STAGES = "):
                                         config.index("    TRAIN_INIT = \"random_decoder\"")].strip())
        train = cells["cell-15"]
        header = train[train.index("    last_epoch = EPOCHS"):train.index("        model.train()\n")]
        body = textwrap.dedent(header) + textwrap.dedent("""
            for _ in range(steps_per_epoch):
                lrs.append([group["lr"] for group in optimizer.param_groups])
                optimizer.step()
                scheduler.step()
                sync_lora_lr(optimizer)
            optimizers.add(id(optimizer))
        """).replace("\n", "\n    ", ) .rstrip()
        adapter = torch.nn.Parameter(torch.zeros(1))
        self.ns["LORA_LR_SCALE"] = lora_scale or 1.0
        self.ns["lora_parameters"] = (lambda model: [adapter]) if lora_scale else (lambda model: [])
        epochs = sum(s["epochs"] for s in stages)
        lrs, optimizers, saved = [], set(), None
        model = torch.nn.Module()
        model.decoder = torch.nn.Linear(1, 1)
        for stop in sessions:
            optimizer = torch.optim.AdamW(self.ns["trainable_groups"](model, stages[0]["lr"]), weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stages[0]["epochs"] * steps_per_epoch, eta_min=stages[0]["min_lr"])
            self.ns["sync_lora_lr"](optimizer)
            start_epoch = 0
            if saved:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                start_epoch = saved["epoch"]
            namespace = {"torch": torch, "EPOCHS": epochs, "STOP_AFTER_EPOCH": stop, "FROM_SCRATCH": True,
                         "SCRATCH_STAGES": stages, "start_epoch": start_epoch, "steps_per_epoch": steps_per_epoch,
                         "optimizer": optimizer, "scheduler": scheduler, "model": model,
                         "lora_parameters": self.ns["lora_parameters"], "trainable_groups": self.ns["trainable_groups"],
                         "sync_lora_lr": self.ns["sync_lora_lr"], "LORA_LR_SCALE": self.ns["LORA_LR_SCALE"],
                         "WEIGHT_DECAY": 0.01, "USE_AMP": False,
                         "generator": torch.Generator(), "SEED": 42, "print": lambda *a, **k: None,
                         "lrs": lrs, "optimizers": optimizers}
            exec(body, namespace)
            saved = {"optimizer": namespace["optimizer"].state_dict(), "scheduler": namespace["scheduler"].state_dict(),
                     "epoch": namespace["last_epoch"]}
        return stages, lrs, steps_per_epoch

    def test_scratch_schedule_matches_e1_then_b1(self):
        stages, lrs, steps = self.scratch_schedule([None])
        self.assertEqual([(s["epochs"], s["lr"], s["min_lr"]) for s in stages],
                         [(5, 1e-3, 1e-4), (2, 1e-4, 5e-5), (2, 3e-4, 1e-5)])
        self.assertEqual(len(lrs), 9 * steps)
        starts = [0, 5 * steps, 7 * steps]
        for start, stage in zip(starts, stages):
            n = stage["epochs"] * steps
            expected = [stage["min_lr"] + (stage["lr"] - stage["min_lr"]) * (1 + math.cos(math.pi * k / n)) / 2
                        for k in range(n)]
            for actual, wanted in zip(lrs[start:start + n], expected):
                self.assertAlmostEqual(actual[0], wanted, places=12)

    def test_scratch_sessions_resume_to_identical_schedule(self):
        _, straight, _ = self.scratch_schedule([None])
        _, split, _ = self.scratch_schedule([3, 6, None])
        self.assertEqual(len(split), len(straight))
        for a, b in zip(split, straight):
            self.assertAlmostEqual(a[0], b[0], places=12)

    def test_lora_lr_scale_follows_decoder_schedule_across_sessions(self):
        _, straight, _ = self.scratch_schedule([None], lora_scale=0.1)
        _, split, _ = self.scratch_schedule([3, 6, None], lora_scale=0.1)
        self.assertEqual(len(straight), len(split))
        for (decoder, lora), (decoder_split, lora_split) in zip(straight, split):
            self.assertAlmostEqual(lora, 0.1 * decoder, places=15)
            self.assertAlmostEqual(decoder_split, decoder, places=15)
            self.assertAlmostEqual(lora_split, lora, places=15)

    def test_single_group_without_lora_matches_b1_optimizer(self):
        model = tiny_model()
        groups = self.ns["trainable_groups"](model, 3e-4)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["lr"], 3e-4)
        self.assertEqual(sum(p.numel() for p in groups[0]["params"]), sum(p.numel() for p in model.decoder.parameters()))

    def test_lora_lr_scale_recorded_only_when_changed(self):
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
        self.assertIn("LORA_LR_SCALE = 1.0", cells["cell-04"])
        self.assertIn('**({"lr_scale": LORA_LR_SCALE} if LORA_LR_SCALE != 1.0 else {})', cells["cell-15"])
        self.assertEqual(cells["cell-15"].count("sync_lora_lr(optimizer)"), 3)  # creation, every step, stage switch

    def test_evaluation_skipped_in_intermediate_sessions(self):
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
        for cell_id in ["cell-17", "cell-19", "e4-save-val", "e4-switch-test"]:
            self.assertTrue(cells[cell_id].startswith("if not STOPPED_EARLY:\n"), cell_id)
        for cell_id in ["cell-07-test", "cell-09-test", "cell-17-test", "cell-19-test", "e4-save-test"]:
            self.assertTrue(cells[cell_id].startswith("if not STOPPED_EARLY and RUN_TEST:\n"), cell_id)
        for cell_id in ["cell-07", "cell-09", "cell-15", "cell-21"]:
            self.assertFalse(cells[cell_id].startswith("if not STOPPED_EARLY"), cell_id)
        self.assertIn("FROM_SCRATCH = True", cells["cell-04"])
        self.assertIn('RESUME_CHECKPOINT = "auto"', cells["cell-04"])
        self.assertIn('if RESUME_CHECKPOINT == "auto":', cells["cell-15"])
        self.assertIn("config_signature", cells["cell-15"])

    def test_notebook_patches(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
        self.assertEqual(len(cells), len(notebook["cells"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
        config = cells["cell-04"]
        for line in ['ARM = "E4c"', 'TRAIN_INIT = "e1_checkpoint"', "TRAIN_DECODER = True", "EPOCHS = 2",
                     "BATCH_SIZE = 64", "USE_AMP = True", "EVAL_AMP = False", "REQUIRE_GPU = True", "MULTI_GPU = True",
                     'EVAL_SPLIT = "val"', "LORA_RANK = 8"]:
            self.assertIn(line, config)
        self.assertIn("segment(model", cells["cell-11"])
        self.assertIn("return segment(self.model", cells["cell-15"])
        self.assertIn('"lora": lora_state_dict(model)', cells["cell-15"])
        self.assertIn("lora_parameters(model)", cells["cell-15"])
        self.assertNotIn("WordAttention", json.dumps(cells))
        order = [c["id"] for c in notebook["cells"]]
        self.assertLess(order.index("cell-11"), order.index("e4-module"))
        self.assertLess(order.index("e4-module"), order.index("cell-15"))
        expected_tail = ["cell-19", "e4-save-val-md", "e4-save-val", "e4-switch-test", "cell-07-test",
                         "cell-09-test", "cell-17-test", "cell-19-test", "e4-save-test"]
        start = order.index("cell-19")
        self.assertEqual(order[start:start + len(expected_tail)], expected_tail)
        self.assertLess(order.index("e4-save-test"), order.index("cell-21"))


if __name__ == "__main__":
    unittest.main()
