import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PIL import Image
from clipseg_project.inference import MODEL_REVISION, Segmenter, save_prediction


def main():
    parser = argparse.ArgumentParser(description="CLIPSeg text or visual prompt inference")
    parser.add_argument("--image", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", nargs="+")
    mode.add_argument("--support-image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0,1]")
    image = Image.open(args.image).convert("RGB")
    support = Image.open(args.support_image).convert("RGB") if args.support_image else None
    segmenter = Segmenter(device=args.device, revision=args.revision)
    probabilities, seconds = segmenter.predict(image, prompts=args.text, support_image=support)
    for i, p in enumerate(probabilities):
        save_prediction(image, p, args.output, f"prompt_{i:02}", args.threshold)
    meta = {
        "model": segmenter.model_id, "requested_revision": args.revision,
        "resolved_commit": getattr(segmenter.model.config, "_commit_hash", None),
        "device": str(segmenter.device), "image": str(args.image),
        "image_size": list(image.size), "prompts": args.text,
        "support_image": str(args.support_image) if args.support_image else None,
        "threshold": args.threshold, "forward_resize_seconds": seconds,
        "timing_note": "Single un-warmed run; excludes loading, preprocessing and saving.",
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "Pillow"]},
        "extract_layers": segmenter.model.config.extract_layers,
        "parameter_count": sum(p.numel() for p in segmenter.model.parameters()),
        "decoder_parameter_count": sum(p.numel() for p in segmenter.model.decoder.parameters()),
    }
    (args.output / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
