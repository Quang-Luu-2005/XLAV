"""Verify the raw-phrase E3 arm as the full-test E1 paper-protocol rerun."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

PROJECT = Path(__file__).resolve().parents[1]
CHAIN = PROJECT / "outputs/kaggle-e3-test-20261015/state.json"
OUTPUT = PROJECT / "outputs/e1-paper-test-epoch7-20261016"
ANNOTATION = "c3ffdd0cd1671a20e3ff80e32bc5448264afd3c49ec3c85877dae620e701aa8f"
PAPER = {"miou_0.3": 0.434, "fgiou_0.3": 0.547, "ap": 0.767}


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def recompute(stats, thresholds):
    assert stats.shape == (14354, 4, 49), stats.shape
    np.testing.assert_allclose(thresholds, np.linspace(0, 1, 51)[1:-1], rtol=0, atol=1e-15)
    assert np.issubdtype(stats.dtype, np.integer) and (stats >= 0).all()
    assert (stats.sum(axis=1) == 352 * 352).all()
    tp, fp, fn = (stats[:, i].astype(np.float64) for i in range(3))
    total_tp, total_fp, total_fn = tp.sum(0), fp.sum(0), fn.sum(0)
    valid = total_tp + total_fp > 0
    points = sorted(set(zip((total_tp / (1 + total_tp + total_fn))[valid],
                            (total_tp / (1 + total_tp + total_fp))[valid])))
    recall, precision = (np.array(x) for x in zip(*points))
    index = int(np.argmin(abs(thresholds - 0.3)))
    return {"miou_0.3": float((tp / (1 + tp + fp + fn)).mean(0)[index]),
            "fgiou_0.3": float((total_tp / (1 + total_tp + total_fp + total_fn))[index]),
            "ap": float(simpson(precision, x=recall))}


def collect(state, output):
    assert state["status"] == "completed" and state["phase"] == "full"
    root = Path(state["final_result"]["artifact_dir"]) / "artifacts"
    result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert result["n_queries"] == 14354 and result["same_queries"]
    models, task_ids, selected_hashes = {}, [], []
    for label in ["e1_decoder", "official_pretrained"]:
        folder = root / label
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert summary["split"] == "test" and summary["complete_official_split"]
        assert summary["n_queries"] == 14354 and run["precision"] == "fp32"
        assert run["annotation_sha256"] == ANNOTATION
        assert json.loads((folder / "image_errors.json").read_text()) == []
        decoder = summary["decoder"]
        if label == "e1_decoder":
            assert decoder["epoch"] == 7 and decoder["checkpoint_sha256"] == state["checkpoint_sha256"]
        else:
            assert decoder["source"] == "pretrained"
        with (folder / "sample_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["variant"] == "{phrase}"]
        ids = [r["task_id"] for r in rows]
        assert len(ids) == len(set(ids)) == 14354
        task_ids.append(ids)
        selected_hashes.append(hashlib.sha256((folder / "selected_tasks.json").read_bytes()).hexdigest())
        with np.load(folder / "paper_threshold_stats.npz", allow_pickle=False) as stored:
            key = next(k for k, v in summary["paper_threshold_stats_keys"].items() if v == "{phrase}")
            metrics = recompute(stored[key], stored["thresholds"])
        for key, value in metrics.items():
            assert abs(value - summary["templates"]["{phrase}"]["paper_protocol"][key]) < 1e-12
        models[label] = {"metrics": metrics, "decoder_source": decoder["source"],
            "epoch": decoder.get("epoch"), "checkpoint_sha256": decoder.get("checkpoint_sha256"),
            "delta_vs_paper_percentage_points": {k: 100 * (metrics[k] - PAPER[k]) for k in PAPER}}
    assert task_ids[0] == task_ids[1] and len(set(selected_hashes)) == 1
    evidence = {"status": "completed", "source_job_id": state["job_id"],
        "source_kernel_slug": state["kernel_slug"], "source_experiment": "E3 raw phrase baseline",
        "split": "test", "n_queries": 14354, "prompt_template": "{phrase}", "threshold": 0.3,
        "resolution": [352, 352], "precision": "fp32", "annotation_sha256": ANNOTATION,
        "paper_reference": {"table": 3, "model": "CLIPSeg (PC+)", "metrics": PAPER,
            "url": "https://arxiv.org/html/2112.10003v2#S5.T3"},
        "models": models, "result_archive_sha256": state["final_result"]["archive_sha256"],
        "delta_e1_minus_pretrained_percentage_points": {
            k: 100 * (models["e1_decoder"]["metrics"][k] - models["official_pretrained"]["metrics"][k]) for k in PAPER}}
    atomic_json(output / "comparison.json", evidence)
    table = ["# E1 versus CLIPSeg paper — complete PhraseCut test", "",
        "14,354 queries; raw phrase; FP32; 352×352; threshold 0.3.", "",
        "| Model | mIoU | IoU_FG | AP |", "|---|---:|---:|---:|"]
    for label, metrics in [("Paper CLIPSeg (PC+)", PAPER),
            ("Official pretrained rerun", models["official_pretrained"]["metrics"]),
            ("E1 epoch 7 rerun", models["e1_decoder"]["metrics"])]:
        table.append(f"| {label} | " + " | ".join(f"{100 * metrics[k]:.2f}" for k in PAPER) + " |")
    table += ["", "Both reruns use identical test queries. Metrics were recomputed from saved pixel counts.",
        "The paper and Hugging Face refined model retain implementation/training differences.",
        "", f"Source job: `{state['job_id']}`. No test-based checkpoint or prompt selection.", ""]
    (output / "comparison.md").write_text("\n".join(table), encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--chain", type=Path, default=CHAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (args.output / "collector.lock").open("a+b") as lock:
        lock.write(b"0"); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        while True:
            state = json.loads(args.chain.read_text(encoding="utf-8"))
            status = {"status": "waiting_for_full_test", "source_job_id": state["job_id"],
                "source_phase": state["phase"], "source_status": state["status"],
                "output": str(args.output)}
            if state["status"] == "completed" and state["phase"] == "full":
                try:
                    evidence = collect(state, args.output)
                except Exception as error:
                    atomic_json(args.output / "state.json", {**status, "status": "verification_failed",
                        "error_type": type(error).__name__, "error": str(error)})
                    raise
                atomic_json(args.output / "state.json", {**status, "status": "completed"})
                print(json.dumps(evidence), flush=True)
                return
            if state["status"] == "stopped":
                atomic_json(args.output / "state.json", {**status, "status": "stopped", "error": state.get("error")})
                print(json.dumps(status), flush=True)
                return
            atomic_json(args.output / "state.json", status)
            print(json.dumps(status), flush=True)
            if not args.watch:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
