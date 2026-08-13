"""Reviewer 2 Comment 3: controlled TA-GAN inference benchmark.

The benchmark times only trajectory prediction from an already prepared
20-point history. It deliberately excludes LiDAR acquisition, tracking, ROS,
file I/O, visualization, and downstream planning.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E2_RUNNER = REPOSITORY_ROOT / "experiments" / "cvm_e2" / "run_cvm_comparison.py"
E3_RUNNER = (
    REPOSITORY_ROOT
    / "experiments"
    / "stochastic_e3"
    / "run_stochastic_evaluation.py"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "ta_gan" / "scripts" / "best_model_indoor.pt"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E2 = load_module("efficiency_e2_helpers", E2_RUNNER)
E3 = load_module("efficiency_e3_helpers", E3_RUNNER)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def make_inputs(batch_size, obs_len, noise_dim, device):
    """Create fixed, non-degenerate histories outside the timed region."""
    steps = torch.arange(obs_len, device=device, dtype=torch.float32)
    offsets = torch.arange(batch_size, device=device, dtype=torch.float32)
    obs = torch.empty(obs_len, batch_size, 2, device=device)
    obs[:, :, 0] = steps[:, None] * 0.015 + offsets[None, :] * 0.01
    obs[:, :, 1] = steps[:, None] * -0.008 + offsets[None, :] * 0.005
    obs_rel = torch.zeros_like(obs)
    obs_rel[1:] = obs[1:] - obs[:-1]
    generator = torch.Generator(device=device)
    generator.manual_seed(20260812 + batch_size)
    noise = torch.randn(
        batch_size, obs_len, noise_dim, generator=generator, device=device
    )
    return obs, obs_rel, noise


def ta_gan_prediction(model, obs, obs_rel, noise):
    """Predict absolute positions for independent one-agent histories."""
    pred_rel = E3.forward_independent_one_agent_groups(
        model, obs, obs_rel, noise
    )
    return pred_rel.cumsum(dim=0) + obs[-1].unsqueeze(0)


def cvm_last_prediction(obs, pred_len):
    """Last-step constant-velocity prediction used by E2."""
    velocity = obs[-1] - obs[-2]
    steps = torch.arange(
        1, pred_len + 1, device=obs.device, dtype=obs.dtype
    ).view(pred_len, 1, 1)
    return obs[-1].unsqueeze(0) + steps * velocity.unsqueeze(0)


def summarize_ms(values):
    values = [float(value) for value in values]
    ordered = sorted(values)
    p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "std_ms": float(statistics.pstdev(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p95_ms": float(ordered[p95_index]),
    }


def benchmark_callable(function, device, warmup, trials, repetitions):
    """Return block-amortized wall time and CUDA device time per call."""
    with torch.no_grad():
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        wall_ms = []
        device_ms = []
        for _ in range(trials):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            wall_start = time.perf_counter()
            for _ in range(repetitions):
                output = function()
            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize(device)
            wall_elapsed = (time.perf_counter() - wall_start) * 1000.0
            wall_ms.append(wall_elapsed / repetitions)
            if device.type == "cuda":
                device_ms.append(
                    start_event.elapsed_time(end_event) / repetitions
                )
            # Materialize one scalar so the output remains live on CPU too.
            _ = float(output.reshape(-1)[0].item())

    result = {"wall_clock": summarize_ms(wall_ms)}
    if device_ms:
        result["cuda_event"] = summarize_ms(device_ms)
    return result, wall_ms, device_ms


def benchmark_device(
    device_name, checkpoint, batch_sizes, obs_len, pred_len,
    warmup, trials, repetitions
):
    device = torch.device(device_name)
    model = E2.load_indoor_generator(checkpoint, device, obs_len)
    total_parameters, trainable_parameters = count_parameters(model)
    rows = []
    raw_trials = []
    for batch_size in batch_sizes:
        obs, obs_rel, noise = make_inputs(
            batch_size, obs_len, model.noise_dim, device
        )
        methods = {
            "TA-GAN": lambda: ta_gan_prediction(
                model, obs, obs_rel, noise
            ),
            "CVM-last": lambda: cvm_last_prediction(obs, pred_len),
        }
        for method, function in methods.items():
            timing, wall_values, event_values = benchmark_callable(
                function, device, warmup, trials, repetitions
            )
            row = {
                "device": device_name,
                "method": method,
                "batch_size": batch_size,
                "parameters": trainable_parameters if method == "TA-GAN" else 0,
                "wall_mean_ms": timing["wall_clock"]["mean_ms"],
                "wall_median_ms": timing["wall_clock"]["median_ms"],
                "wall_std_ms": timing["wall_clock"]["std_ms"],
                "wall_p95_ms": timing["wall_clock"]["p95_ms"],
                "cuda_mean_ms": timing.get("cuda_event", {}).get("mean_ms"),
                "cuda_median_ms": timing.get("cuda_event", {}).get("median_ms"),
                "cuda_std_ms": timing.get("cuda_event", {}).get("std_ms"),
                "cuda_p95_ms": timing.get("cuda_event", {}).get("p95_ms"),
                "throughput_histories_per_s": batch_size * 1000.0
                / timing["wall_clock"]["mean_ms"],
            }
            rows.append(row)
            for trial_index, value in enumerate(wall_values):
                raw_trials.append({
                    "device": device_name,
                    "method": method,
                    "batch_size": batch_size,
                    "timer": "wall_clock",
                    "trial": trial_index,
                    "latency_ms_per_call": value,
                })
            for trial_index, value in enumerate(event_values):
                raw_trials.append({
                    "device": device_name,
                    "method": method,
                    "batch_size": batch_size,
                    "timer": "cuda_event",
                    "trial": trial_index,
                    "latency_ms_per_call": value,
                })
    return rows, raw_trials, total_parameters, trainable_parameters


def write_csv(path, rows):
    if not rows:
        return
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    devices = ["cpu"]
    if args.device in ("auto", "cuda"):
        if torch.cuda.is_available():
            devices.insert(0, "cuda")
        elif args.device == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
    rows = []
    raw_trials = []
    parameter_counts = set()
    for device_name in devices:
        device_rows, device_trials, total, trainable = benchmark_device(
            device_name,
            checkpoint,
            args.batch_sizes,
            args.obs_len,
            args.pred_len,
            args.warmup,
            args.trials,
            args.repetitions,
        )
        rows.extend(device_rows)
        raw_trials.extend(device_trials)
        parameter_counts.add((total, trainable))
    if len(parameter_counts) != 1:
        raise AssertionError("parameter count changed across devices")
    total_parameters, trainable_parameters = parameter_counts.pop()

    write_csv(output_dir / "latency_summary.csv", rows)
    write_csv(output_dir / "latency_trials.csv", raw_trials)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E4_R2_3_CONTROLLED_INFERENCE_EFFICIENCY",
        "scope": (
            "Prediction tensor computation only; excludes LiDAR, tracking, "
            "ROS, I/O, visualization, and planning."
        ),
        "grouping": (
            "Each batch item is an independent one-agent history. This does "
            "not measure multi-agent quadratic interaction scaling."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "generator_total_parameters": total_parameters,
        "generator_trainable_parameters": trainable_parameters,
        "cvm_trainable_parameters": 0,
        "arguments": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu": args.cpu_model or platform.processor(),
            "cpu_threads_used": args.cpu_threads,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
            if torch.cuda.is_available() else None,
        },
        "results": rows,
        "limitations": [
            "Latency is machine- and software-version-specific.",
            "The benchmark uses fixed prepared tensors and batch-amortized timing.",
            "CVM is a zero-trainable-parameter arithmetic baseline.",
            "No closed-loop navigation latency or safety is measured.",
            "Independent one-agent histories do not characterize multi-agent scaling.",
        ],
    }
    with open(str(output_dir / "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    artifacts = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_sha256.csv":
            artifacts.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    write_csv(output_dir / "artifact_sha256.csv", artifacts)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--cpu-model", default="")
    parser.add_argument("--seed", type=int, default=20260812)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
