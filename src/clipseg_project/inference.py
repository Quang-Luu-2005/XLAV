"""CLIPSeg text and image-conditioned inference, without per-image normalization."""
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

MODEL_ID = "CIDAS/clipseg-rd64-refined"
MODEL_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"


class Segmenter:
    def __init__(self, device="auto", model_id=MODEL_ID, revision=MODEL_REVISION):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = CLIPSegProcessor.from_pretrained(model_id, revision=revision, use_fast=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_id, revision=revision
        ).to(self.device).eval()
        self.model_id = model_id
        self.revision = revision

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, image, prompts=None, support_image=None):
        """Return [N,H,W] sigmoid probabilities and forward+resize seconds.

        support_image should come from a different image from the query for E2.
        Passing a crop from the query is allowed for a demo, but is a different protocol.
        """
        if (prompts is None) == (support_image is None):
            raise ValueError("Supply either text prompts or a support image.")
        image = image.convert("RGB")
        if support_image is None:
            if not prompts or any(not p.strip() for p in prompts):
                raise ValueError("Text prompts must be nonempty.")
            inputs = dict(self.processor.tokenizer(
                text=prompts,
                padding=True, truncation=True, return_tensors="pt",
            ))
            inputs.update(self.processor(images=[image] * len(prompts), return_tensors="pt"))
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            support = self.processor(images=support_image.convert("RGB"), return_tensors="pt")
            inputs["conditional_pixel_values"] = support["pixel_values"]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self._sync()
        start = time.perf_counter()
        logits = self.model(**inputs).logits
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        logits = F.interpolate(
            logits.unsqueeze(1), size=(image.height, image.width),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        probabilities = logits.sigmoid()
        self._sync()
        seconds = time.perf_counter() - start
        return probabilities.cpu().numpy(), seconds


def save_prediction(image, probability, directory, stem, threshold=0.3):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{stem}_probability.npy", probability)
    binary = probability >= threshold
    Image.fromarray(binary.astype(np.uint8) * 255).save(directory / f"{stem}_mask.png")
    Image.fromarray((probability * 255).round().astype(np.uint8)).save(
        directory / f"{stem}_heatmap.png"
    )
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    rgb[binary] = 0.55 * rgb[binary] + 0.45 * np.array([255, 60, 60])
    Image.fromarray(rgb.round().astype(np.uint8)).save(directory / f"{stem}_overlay.png")
