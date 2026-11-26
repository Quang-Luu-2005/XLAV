"""Preserve every training sample; diagnose AMP failures and retry in FP32.

Embedded in the Kaggle notebook by build_e4_notebook.py. No model is run here.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F


NUMERICAL_POLICY = "amp_with_verified_fp32_batch_retry_v1"


def tensor_summary(value):
    values = []
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        finite = torch.isfinite(value)
        values.append({"shape": list(value.shape), "dtype": str(value.dtype),
                       "finite": bool(finite.all()),
                       "nonfinite": int((~finite).sum()),
                       "finite_abs_max": float(value.detach()[finite].abs().max()) if finite.any() else None})
    elif isinstance(value, (tuple, list, dict)):
        for item in value.values() if isinstance(value, dict) else value:
            values.extend(tensor_summary(item))
    return values


def save_numerical_report(directory, report):
    path = Path(directory) / "numerical_recovery.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def loss_backward_with_recovery(forward, model, inputs, labels, scaler, directory,
                                epoch, batch_index, global_step, microbatch_size=16):
    """One effective batch and one optimizer step, including a rare FP32 retry.

    The caller owns optimizer.zero_grad/step and the scheduler. FP32 microbatch
    losses are weighted by sample count before backward, so batch 64 stays 64.
    A nonfinite FP32 loss raises before any optimizer step. No batch is dropped.
    """
    logits = forward(**inputs)
    loss = F.binary_cross_entropy_with_logits(logits.float(), labels.to(logits.device))
    if torch.isfinite(loss):
        scaler.scale(loss).backward()
        return loss.detach()

    # Release the invalid graph before the diagnostic replay and FP32 backward.
    logits, loss = None, None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "numerical_recovery.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "policy": NUMERICAL_POLICY, "events": []}
    event = {"epoch": epoch, "batch_index": batch_index, "global_step": global_step,
             "effective_batch_size": len(labels), "fp32_microbatch_size": microbatch_size,
             "inputs": {key: tensor_summary(value) for key, value in inputs.items()},
             "labels": tensor_summary(labels), "first_nonfinite_modules": []}
    report["events"].append(event)
    # Always retain the first failing batch and its model weights for GPU replay.
    if len(report["events"]) == 1:
        torch.save({"inputs": {k: v.detach().cpu() for k, v in inputs.items()},
                    "labels": labels.detach().cpu(),
                    "trainable_state": {k: p.detach().cpu() for k, p in model.named_parameters() if p.requires_grad},
                    "epoch": epoch, "batch_index": batch_index, "global_step": global_step},
                   directory / "first_nonfinite_batch.pt")
    tensors = [t for t in inputs.values() if t.is_floating_point()] + [labels]
    if not all(torch.isfinite(t).all() for t in tensors):
        event["classification"] = "nonfinite_input_or_label"
        save_numerical_report(directory, report)
        raise ValueError("Nonfinite training input/label; diagnostic saved, no update")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        event["classification"] = "nonfinite_model_parameters"
        save_numerical_report(directory, report)
        raise ValueError("Nonfinite model parameters; diagnostic saved, no update")

    def hook(name):
        def inspect(module, args, output):
            # First failing leaf/compound on each device. All hooks are removed
            # before backward so a graph never retains diagnostic tensors.
            if len(event["first_nonfinite_modules"]) < 6:
                summaries = tensor_summary(output)
                if any(not item["finite"] for item in summaries):
                    event["first_nonfinite_modules"].append({"name": name,
                        "type": type(module).__name__, "inputs": tensor_summary(args), "outputs": summaries})
        return inspect

    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()]
    try:
        with torch.no_grad():
            replay = forward(**inputs)
            event["amp_replay_finite"] = bool(torch.isfinite(replay).all())
            del replay
    finally:
        for handle in handles:
            handle.remove()
    event["classification"] = "awaiting_fp32_same_batch_check"
    save_numerical_report(directory, report)
    print("AMP_NONFINITE_DIAGNOSTIC " + json.dumps(event), flush=True)

    total, result_device = 0.0, labels.device
    try:
        for start in range(0, len(labels), microbatch_size):
            end = min(start + microbatch_size, len(labels))
            chunk = {key: value[start:end] for key, value in inputs.items()}
            logits = forward(**chunk, force_fp32=True)
            result_device = logits.device
            part = F.binary_cross_entropy_with_logits(logits.float(), labels[start:end].to(logits.device))
            if not torch.isfinite(part):
                raise ValueError("Same batch is nonfinite in FP32 too; no optimizer update")
            weight = (end - start) / len(labels)
            scaler.scale(part * weight).backward()
            total += float(part.detach()) * weight
            del logits, part
    except Exception:
        model.zero_grad(set_to_none=True)
        event["classification"] = "fp32_retry_failed"
        save_numerical_report(directory, report)
        raise
    event.update(classification="fp16_numerical_failure_fp32_same_batch_finite", fp32_loss=total,
                 backward_completed=True, samples_skipped=0)
    save_numerical_report(directory, report)
    print("AMP_RECOVERED " + json.dumps(event), flush=True)
    return torch.tensor(total, device=result_device)
