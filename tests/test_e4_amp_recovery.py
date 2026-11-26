"""Tiny CPU arithmetic tests; real CLIPSeg diagnosis only runs on Kaggle."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e4_amp_recovery import loss_backward_with_recovery


class ToyForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(2)

    def forward(self, x, force_fp32=False):
        with torch.autocast("cpu", dtype=torch.float16, enabled=not force_fp32):
            return self.linear(x).squeeze(-1)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def test_same_samples_and_weighted_gradients_after_fp16_overflow(self):
        model = ToyForward()
        inputs = {"x": torch.tensor([[20000., 20000.], [21000., 20000.], [22000., 20000.],
                                     [23000., 20000.], [24000., 20000.]])}
        labels = torch.zeros(5)
        reference = F.binary_cross_entropy_with_logits(model(**inputs, force_fp32=True), labels)
        reference.backward()
        expected_grad = model.linear.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as temporary:
            actual = loss_backward_with_recovery(model, model, inputs, labels, self.scaler,
                temporary, 2, 1739, 6594, microbatch_size=2)
            report = json.loads((Path(temporary) / "numerical_recovery.json").read_text())
            event = report["events"][0]
            self.assertEqual(event["classification"], "fp16_numerical_failure_fp32_same_batch_finite")
            self.assertEqual(event["samples_skipped"], 0)
            self.assertEqual(event["effective_batch_size"], 5)
            self.assertEqual(event["first_nonfinite_modules"][0]["name"], "linear")
            saved = torch.load(Path(temporary) / "first_nonfinite_batch.pt", weights_only=True)
            torch.testing.assert_close(saved["inputs"]["x"], inputs["x"])
        torch.testing.assert_close(actual, reference.detach())
        torch.testing.assert_close(model.linear.weight.grad, expected_grad)

    def test_normal_batch_does_not_create_diagnostics(self):
        model = ToyForward()
        with tempfile.TemporaryDirectory() as temporary:
            result = loss_backward_with_recovery(model, model, {"x": torch.ones(3, 2)},
                torch.zeros(3), self.scaler, temporary, 1, 0, 0)
            self.assertTrue(torch.isfinite(result))
            self.assertTrue(torch.isfinite(model.linear.weight.grad).all())
            self.assertFalse(list(Path(temporary).iterdir()))

    def test_nonfinite_input_stops_instead_of_skipping(self):
        model = ToyForward()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "input/label"):
                loss_backward_with_recovery(model, model, {"x": torch.full((2, 2), float("nan"))},
                    torch.zeros(2), self.scaler, temporary, 1, 0, 0)
            self.assertIsNone(model.linear.weight.grad)

    def test_fp32_failure_clears_partial_gradients(self):
        model = ToyForward()

        def bad_forward(x, force_fp32=False):
            result = model(x, force_fp32=force_fp32)
            return result * float("nan") if force_fp32 and x[0, 0] > 22000 else result

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "FP32 too"):
                loss_backward_with_recovery(bad_forward, model,
                    {"x": torch.tensor([[20000., 20000.], [24000., 20000.]])},
                    torch.zeros(2), self.scaler, temporary, 1, 0, 0, microbatch_size=1)
            self.assertIsNone(model.linear.weight.grad)
            report = json.loads((Path(temporary) / "numerical_recovery.json").read_text())
            self.assertEqual(report["events"][0]["classification"], "fp32_retry_failed")


if __name__ == "__main__":
    unittest.main()
