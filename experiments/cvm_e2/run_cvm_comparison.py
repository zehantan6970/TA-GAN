"""Reproducible CVM comparison for Reviewer 1.3 / experiment E2.

The retained IndoorNav files contain one trajectory per text file with three
columns: timestamp (microseconds), x (metres), and y (metres).  The original
paper's random 8:2 split manifest is unavailable, so this runner implements an
explicit reconstructed protocol and records that limitation in every output.

Primary protocol
----------------
* 20 observed points and 20 predicted points.
* Sliding windows with stride 10 (configurable).
* Deterministic 20% scene subset selected by SHA-256 for ``--scope hash20``.
* CVM-last: velocity from the final two observed points.
* CVM-LS: linear least-squares fit over all observed points.
* Actual timestamps are used by both CVM variants; no fixed-rate assumption.
* ADE/FDE are Euclidean distances in metres.
* Confidence intervals use a source-file cluster bootstrap, retaining all
  windows from a sampled trajectory file together.

TA-GAN is evaluated on the identical windows.  Each raw trajectory is an
independent one-agent group because the original multi-agent preprocessing and
group manifest were not retained.  Checkpoint training exposure is unknown and
must not be described as a leakage-free model evaluation.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "ta_gan"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sgan.models_transformer_ori import Trajectory_Generator  # noqa: E402


@dataclass
class TrajectoryRecord:
    relative_path: str
    scene: str
    timestamps_us: np.ndarray
    positions: np.ndarray
    sha256: str
    size_bytes: int


@dataclass
class Window:
    sample_id: str
    source_file: str
    scene: str
    start_index: int
    timestamps_s: np.ndarray
    positions: np.ndarray


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_is_in_hash20(scene):
    """Return the frozen, deterministic 20% scene assignment."""
    digest = hashlib.sha256(scene.lower().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % 5 == 0


def read_trajectory(path, dataset_root):
    """Parse and validate one retained ``timestamp x y`` trajectory file."""
    if path.stat().st_size == 0:
        raise ValueError("file is empty")
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("expected exactly three numeric columns")
    if not np.isfinite(data).all():
        raise ValueError("contains non-finite values")
    timestamps = data[:, 0]
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps are not strictly increasing")

    relative = path.relative_to(dataset_root)
    scene = relative.parts[0]
    return TrajectoryRecord(
        relative_path=relative.as_posix(),
        scene=scene,
        timestamps_us=timestamps,
        positions=data[:, 1:3],
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def discover_records(dataset_root, scope="hash20", max_files=None):
    """Discover numeric trajectories and return records plus parse failures."""
    paths = sorted(dataset_root.rglob("*.txt"), key=lambda item: item.as_posix())
    records = []
    failures = []
    for path in paths:
        relative = path.relative_to(dataset_root)
        if not relative.parts or not relative.parts[0].lower().startswith("scene"):
            continue
        if scope == "hash20" and not scene_is_in_hash20(relative.parts[0]):
            continue
        if max_files is not None and len(records) >= max_files:
            break
        try:
            records.append(read_trajectory(path, dataset_root))
        except (OSError, ValueError) as error:
            failures.append(
                {"relative_path": relative.as_posix(), "error": str(error)}
            )
    return records, failures


def make_windows(record, obs_len, pred_len, stride):
    """Create fixed-length, overlapping windows from one source trajectory."""
    seq_len = obs_len + pred_len
    windows = []
    for start in range(0, len(record.timestamps_us) - seq_len + 1, stride):
        end = start + seq_len
        timestamps_s = (
            record.timestamps_us[start:end] - record.timestamps_us[start]
        ) / 1_000_000.0
        sample_id = "{}#{}".format(record.relative_path, start)
        windows.append(
            Window(
                sample_id=sample_id,
                source_file=record.relative_path,
                scene=record.scene,
                start_index=start,
                timestamps_s=timestamps_s,
                positions=record.positions[start:end].copy(),
            )
        )
    return windows


def cvm_last_two(obs_t, obs_pos, future_t):
    """Constant velocity from the final two observations."""
    delta_t = obs_t[-1] - obs_t[-2]
    if delta_t <= 0:
        raise ValueError("the last two observation timestamps must increase")
    velocity = (obs_pos[-1] - obs_pos[-2]) / delta_t
    return obs_pos[-1] + np.outer(future_t - obs_t[-1], velocity)


def cvm_least_squares(obs_t, obs_pos, future_t):
    """Fit x(t), y(t) jointly over every observed point and extrapolate."""
    centred_t = obs_t - obs_t[-1]
    design = np.column_stack([centred_t, np.ones_like(centred_t)])
    coefficients, _, _, _ = np.linalg.lstsq(design, obs_pos, rcond=None)
    future_design = np.column_stack(
        [future_t - obs_t[-1], np.ones_like(future_t)]
    )
    return future_design.dot(coefficients)


def displacement_metrics(prediction, target):
    distances = np.linalg.norm(prediction - target, axis=-1)
    return float(np.mean(distances)), float(distances[-1])


def heading_change_degrees(positions, obs_len):
    """Ground-truth heading change used only for post-hoc regime analysis."""
    span = min(5, obs_len - 1, len(positions) - obs_len - 1)
    before = positions[obs_len - 1] - positions[obs_len - 1 - span]
    after = positions[-1] - positions[-1 - span]
    before_norm = np.linalg.norm(before)
    after_norm = np.linalg.norm(after)
    if before_norm < 0.02 or after_norm < 0.02:
        return float("nan")
    cosine = np.dot(before, after) / (before_norm * after_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def motion_regime(positions, obs_len):
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    angle = heading_change_degrees(positions, obs_len)
    if path_length < 0.10 or math.isnan(angle):
        return "low_motion", angle, path_length
    if angle <= 10.0:
        return "straight", angle, path_length
    if angle >= 20.0:
        return "turning", angle, path_length
    return "transition", angle, path_length


def load_indoor_generator(checkpoint_path, device, obs_len):
    model = Trajectory_Generator(
        obs_len=obs_len,
        embedding_dim=16,
        encoder_input_dim=16,
        encoder_output_dim=16,
        encoder_mlp_dim=16,
        encoder_num_head=2,
        drop_rate=0,
        rel_traj_dim=16,
        noise_dim=4,
        merge_mlp_dim=16,
    )
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def predict_ta_gan(model, windows, obs_len, pred_len, samples, batch_size, seed, device):
    """Return deterministic zero-noise and K-sample stochastic predictions."""
    count = len(windows)
    targets = np.stack([item.positions[obs_len:] for item in windows])
    zero_predictions = np.empty((count, pred_len, 2), dtype=np.float32)
    first_predictions = np.empty_like(zero_predictions)
    best_ade_predictions = np.empty_like(zero_predictions)
    ade_samples = np.empty((count, samples), dtype=np.float64)
    fde_samples = np.empty((count, samples), dtype=np.float64)
    best_seen = np.full(count, np.inf, dtype=np.float64)

    def run_batch(batch_windows, noise_array):
        obs_np = np.stack(
            [item.positions[:obs_len] for item in batch_windows], axis=1
        ).astype(np.float32)
        obs = torch.from_numpy(obs_np).to(device)
        obs_rel = torch.zeros_like(obs)
        obs_rel[1:] = obs[1:] - obs[:-1]
        groups = torch.arange(
            len(batch_windows) + 1, device=device, dtype=torch.long
        )
        groups = torch.stack([groups[:-1], groups[1:]], dim=1)
        noise = torch.from_numpy(noise_array).to(device)
        with torch.no_grad():
            pred_rel = model(obs, obs_rel, groups, noise=noise)
            pred_abs = pred_rel.cumsum(dim=0) + obs[-1].unsqueeze(0)
        return pred_abs.transpose(0, 1).cpu().numpy()

    zero_noise = np.zeros((batch_size, obs_len, 4), dtype=np.float32)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        zero_predictions[start:end] = run_batch(
            windows[start:end], zero_noise[: end - start]
        )

    for sample_index in range(samples):
        random = np.random.RandomState(seed + sample_index)
        all_noise = random.standard_normal((count, obs_len, 4)).astype(np.float32)
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            prediction = run_batch(windows[start:end], all_noise[start:end])
            if sample_index == 0:
                first_predictions[start:end] = prediction
            distances = np.linalg.norm(
                prediction - targets[start:end], axis=-1
            )
            ade = distances.mean(axis=1)
            ade_samples[start:end, sample_index] = ade
            fde_samples[start:end, sample_index] = distances[:, -1]
            improved = ade < best_seen[start:end]
            if np.any(improved):
                global_indices = np.arange(start, end)[improved]
                best_ade_predictions[global_indices] = prediction[improved]
                best_seen[global_indices] = ade[improved]

    return {
        "zero_predictions": zero_predictions,
        "first_predictions": first_predictions,
        "best_ade_predictions": best_ade_predictions,
        "zero_ade": np.linalg.norm(zero_predictions - targets, axis=-1).mean(axis=1),
        "zero_fde": np.linalg.norm(zero_predictions[:, -1] - targets[:, -1], axis=-1),
        "mean_ade": ade_samples.mean(axis=1),
        "mean_fde": fde_samples.mean(axis=1),
        "min_ade": ade_samples.min(axis=1),
        "min_fde": fde_samples.min(axis=1),
    }


def cluster_bootstrap(values, clusters, iterations, seed):
    """Compute a file-cluster bootstrap CI for a window-weighted mean."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    cluster_sums = np.asarray([values[clusters == item].sum() for item in unique])
    cluster_counts = np.asarray([(clusters == item).sum() for item in unique])
    random = np.random.RandomState(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selection = random.randint(0, len(unique), size=len(unique))
        estimates[index] = cluster_sums[selection].sum() / cluster_counts[
            selection
        ].sum()
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def summarize_metric(values, clusters, bootstrap_iterations, seed):
    return {
        "mean": float(np.mean(values)),
        "ci95_cluster_bootstrap": cluster_bootstrap(
            values, clusters, bootstrap_iterations, seed
        ),
    }


def summarize_comparison(ta_values, cvm_values, clusters, iterations, seed):
    difference = np.asarray(ta_values) - np.asarray(cvm_values)
    return {
        "mean_difference_ta_minus_cvm": float(np.mean(difference)),
        "ci95_difference": cluster_bootstrap(
            difference, clusters, iterations, seed
        ),
        "ta_gan_better_window_fraction": float(np.mean(difference < 0)),
        "ties_window_fraction": float(np.mean(difference == 0)),
    }


def write_csv(path, rows, fieldnames):
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_paired_ade(output_path, rows, samples):
    cvm_last = np.asarray([row["cvm_last_ade"] for row in rows])
    cvm_ls = np.asarray([row["cvm_ls_ade"] for row in rows])
    ta = np.asarray([row["ta_mean_ade"] for row in rows])
    regimes = sorted(set(row["regime"] for row in rows))
    last_differences = [
        np.asarray(
            [
                row["ta_mean_ade"] - row["cvm_last_ade"]
                for row in rows
                if row["regime"] == regime
            ]
        )
        for regime in regimes
    ]
    ls_differences = [
        np.asarray(
            [
                row["ta_mean_ade"] - row["cvm_ls_ade"]
                for row in rows
                if row["regime"] == regime
            ]
        )
        for regime in regimes
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, cvm, name in (
        (axes[0], cvm_last, "CVM-last"),
        (axes[1], cvm_ls, "CVM-LS"),
    ):
        axis.scatter(cvm, ta, s=8, alpha=0.22, color="#2b6f77")
        limit = max(float(np.percentile(np.concatenate([cvm, ta]), 99)), 0.05)
        axis.plot([0, limit], [0, limit], color="#8f2d2d", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_xlabel("{} ADE (m)".format(name))
        axis.set_ylabel("TA-GAN mean@{} ADE (m)".format(samples))
        axis.set_title("TA-GAN vs {}".format(name))

    positions = np.arange(len(regimes))
    last_boxes = axes[2].boxplot(
        last_differences,
        positions=positions - 0.18,
        widths=0.30,
        showfliers=False,
        patch_artist=True,
    )
    ls_boxes = axes[2].boxplot(
        ls_differences,
        positions=positions + 0.18,
        widths=0.30,
        showfliers=False,
        patch_artist=True,
    )
    for box in last_boxes["boxes"]:
        box.set_facecolor("#c44e52")
        box.set_alpha(0.55)
    for box in ls_boxes["boxes"]:
        box.set_facecolor("#2b6f77")
        box.set_alpha(0.55)
    axes[2].axhline(0, color="#202020", linewidth=1)
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(regimes, rotation=20)
    axes[2].set_ylabel("TA-GAN - CVM ADE (m)")
    axes[2].set_title("Difference by motion regime")
    axes[2].legend(
        handles=[
            Patch(facecolor="#c44e52", alpha=0.55, label="CVM-last"),
            Patch(facecolor="#2b6f77", alpha=0.55, label="CVM-LS"),
        ],
        loc="lower left",
    )
    figure.tight_layout()
    figure.savefig(str(output_path), dpi=180)
    plt.close(figure)


def choose_qualitative_indices(rows):
    chosen = []
    priorities = ["straight", "turning", "low_motion", "transition"]
    for regime in priorities:
        indices = [index for index, row in enumerate(rows) if row["regime"] == regime]
        if not indices:
            continue
        differences = np.asarray(
            [rows[index]["ta_mean_ade"] - rows[index]["cvm_last_ade"] for index in indices]
        )
        median = np.median(differences)
        chosen.append(indices[int(np.argmin(np.abs(differences - median)))])
        if len(chosen) == 4:
            break
    return chosen


def plot_qualitative(
    output_path, rows, windows, cvm_predictions, ta_outputs, obs_len, samples
):
    indices = choose_qualitative_indices(rows)
    if not indices:
        return
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for axis, index in zip(axes.ravel(), indices):
        positions = windows[index].positions
        axis.plot(
            positions[:obs_len, 0], positions[:obs_len, 1], "o-", ms=2.5,
            color="#3b4a6b", label="Observed"
        )
        axis.plot(
            positions[obs_len:, 0], positions[obs_len:, 1], "o-", ms=2.5,
            color="#202020", label="Ground truth"
        )
        axis.plot(
            cvm_predictions[index, :, 0], cvm_predictions[index, :, 1], "--",
            color="#c44e52", label="CVM-last"
        )
        axis.plot(
            ta_outputs["first_predictions"][index, :, 0],
            ta_outputs["first_predictions"][index, :, 1],
            "--", color="#2b6f77", label="TA-GAN fixed sample"
        )
        axis.plot(
            ta_outputs["best_ade_predictions"][index, :, 0],
            ta_outputs["best_ade_predictions"][index, :, 1],
            ":", color="#6c4f8b", label="TA-GAN best-of-{} (oracle)".format(samples)
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title(
            "{}\nCVM-last {:.3f}, TA mean@{} {:.3f}".format(
                rows[index]["regime"],
                rows[index]["cvm_last_ade"],
                samples,
                rows[index]["ta_mean_ade"],
            ),
            fontsize=10,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=[0, 0.08, 1, 1])
    figure.savefig(str(output_path), dpi=180)
    plt.close(figure)


def dataset_fingerprint(records):
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(record.sha256.encode("ascii"))
    return digest.hexdigest()


def run(args):
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, failures = discover_records(
        dataset_root, scope=args.scope, max_files=args.max_files
    )
    windows = []
    for record in records:
        windows.extend(
            make_windows(record, args.obs_len, args.pred_len, args.stride)
        )
    if not windows:
        raise RuntimeError("no valid {}-point windows found".format(
            args.obs_len + args.pred_len
        ))

    rows = []
    cvm_last_predictions = []
    cvm_ls_predictions = []
    for window in windows:
        obs_t = window.timestamps_s[: args.obs_len]
        future_t = window.timestamps_s[args.obs_len :]
        obs_pos = window.positions[: args.obs_len]
        target = window.positions[args.obs_len :]
        last_prediction = cvm_last_two(obs_t, obs_pos, future_t)
        ls_prediction = cvm_least_squares(obs_t, obs_pos, future_t)
        last_ade, last_fde = displacement_metrics(last_prediction, target)
        ls_ade, ls_fde = displacement_metrics(ls_prediction, target)
        regime, heading_change, path_length = motion_regime(
            window.positions, args.obs_len
        )
        rows.append(
            {
                "sample_id": window.sample_id,
                "source_file": window.source_file,
                "scene": window.scene,
                "start_index": window.start_index,
                "regime": regime,
                "heading_change_deg": heading_change,
                "path_length_m": path_length,
                "median_dt_s": float(np.median(np.diff(window.timestamps_s))),
                "cvm_last_ade": last_ade,
                "cvm_last_fde": last_fde,
                "cvm_ls_ade": ls_ade,
                "cvm_ls_fde": ls_fde,
            }
        )
        cvm_last_predictions.append(last_prediction)
        cvm_ls_predictions.append(ls_prediction)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_indoor_generator(checkpoint, device, args.obs_len)
    ta_outputs = predict_ta_gan(
        model,
        windows,
        args.obs_len,
        args.pred_len,
        args.samples,
        args.batch_size,
        args.seed,
        device,
    )
    for index, row in enumerate(rows):
        row.update(
            {
                "ta_zero_ade": float(ta_outputs["zero_ade"][index]),
                "ta_zero_fde": float(ta_outputs["zero_fde"][index]),
                "ta_mean_ade": float(ta_outputs["mean_ade"][index]),
                "ta_mean_fde": float(ta_outputs["mean_fde"][index]),
                "ta_min_ade": float(ta_outputs["min_ade"][index]),
                "ta_min_fde": float(ta_outputs["min_fde"][index]),
            }
        )

    clusters = np.asarray([row["source_file"] for row in rows])
    metric_columns = {
        "CVM-last": ("cvm_last_ade", "cvm_last_fde"),
        "CVM-LS": ("cvm_ls_ade", "cvm_ls_fde"),
        "TA-GAN zero-noise": ("ta_zero_ade", "ta_zero_fde"),
        "TA-GAN mean@{}".format(args.samples): ("ta_mean_ade", "ta_mean_fde"),
        "TA-GAN min@{}".format(args.samples): ("ta_min_ade", "ta_min_fde"),
    }
    method_summary = {}
    summary_rows = []
    for method_index, (method, columns) in enumerate(metric_columns.items()):
        ade_values = np.asarray([row[columns[0]] for row in rows])
        fde_values = np.asarray([row[columns[1]] for row in rows])
        ade_summary = summarize_metric(
            ade_values,
            clusters,
            args.bootstrap_iterations,
            args.seed + method_index * 2,
        )
        fde_summary = summarize_metric(
            fde_values,
            clusters,
            args.bootstrap_iterations,
            args.seed + method_index * 2 + 1,
        )
        method_summary[method] = {"ade_m": ade_summary, "fde_m": fde_summary}
        summary_rows.append(
            {
                "method": method,
                "ade_m": ade_summary["mean"],
                "ade_ci95_low": ade_summary["ci95_cluster_bootstrap"][0],
                "ade_ci95_high": ade_summary["ci95_cluster_bootstrap"][1],
                "fde_m": fde_summary["mean"],
                "fde_ci95_low": fde_summary["ci95_cluster_bootstrap"][0],
                "fde_ci95_high": fde_summary["ci95_cluster_bootstrap"][1],
            }
        )

    comparisons = {}
    for cvm_index, (cvm_name, cvm_prefix) in enumerate(
        (("CVM-last", "cvm_last"), ("CVM-LS", "cvm_ls"))
    ):
        for metric_index, metric in enumerate(("ade", "fde")):
            key = "TA-GAN mean@{} vs {} {}".format(
                args.samples, cvm_name, metric.upper()
            )
            comparisons[key] = summarize_comparison(
                [row["ta_mean_{}".format(metric)] for row in rows],
                [row["{}_{}".format(cvm_prefix, metric)] for row in rows],
                clusters,
                args.bootstrap_iterations,
                args.seed + 50 + cvm_index * 2 + metric_index,
            )

    regime_summary = {}
    for regime in sorted(set(row["regime"] for row in rows)):
        selected = [row for row in rows if row["regime"] == regime]
        regime_summary[regime] = {
            "windows": len(selected),
            "source_files": len(set(row["source_file"] for row in selected)),
            "cvm_last_ade_m": float(
                np.mean([row["cvm_last_ade"] for row in selected])
            ),
            "cvm_ls_ade_m": float(np.mean([row["cvm_ls_ade"] for row in selected])),
            "ta_mean_ade_m": float(
                np.mean([row["ta_mean_ade"] for row in selected])
            ),
            "mean_difference_ta_minus_cvm_last_ade_m": float(
                np.mean(
                    [
                        row["ta_mean_ade"] - row["cvm_last_ade"]
                        for row in selected
                    ]
                )
            ),
            "mean_difference_ta_minus_cvm_ls_ade_m": float(
                np.mean(
                    [
                        row["ta_mean_ade"] - row["cvm_ls_ade"]
                        for row in selected
                    ]
                )
            ),
            "ta_gan_better_than_cvm_last_window_fraction": float(
                np.mean(
                    [
                        row["ta_mean_ade"] < row["cvm_last_ade"]
                        for row in selected
                    ]
                )
            ),
            "ta_gan_better_than_cvm_ls_window_fraction": float(
                np.mean(
                    [
                        row["ta_mean_ade"] < row["cvm_ls_ade"]
                        for row in selected
                    ]
                )
            ),
        }

    manifest_rows = [
        {
            "relative_path": record.relative_path,
            "scene": record.scene,
            "points": len(record.timestamps_us),
            "windows": max(
                0,
                (len(record.timestamps_us) - args.obs_len - args.pred_len)
                // args.stride
                + 1,
            ),
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
        }
        for record in records
    ]
    write_csv(
        output_dir / "dataset_manifest.csv",
        manifest_rows,
        ["relative_path", "scene", "points", "windows", "size_bytes", "sha256"],
    )
    write_csv(output_dir / "per_window_metrics.csv", rows, list(rows[0].keys()))
    write_csv(output_dir / "summary_table.csv", summary_rows, list(summary_rows[0].keys()))
    if failures:
        write_csv(output_dir / "parse_failures.csv", failures, ["relative_path", "error"])

    plot_paired_ade(output_dir / "paired_ade.png", rows, args.samples)
    plot_qualitative(
        output_dir / "qualitative_examples.png",
        rows,
        windows,
        np.asarray(cvm_last_predictions),
        ta_outputs,
        args.obs_len,
        args.samples,
    )

    selected_scenes = sorted(set(record.scene for record in records))
    all_scenes = sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.lower().startswith("scene")
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "RECONSTRUCTED_RETAINED_DATA_PROTOCOL",
        "limitations": [
            "The original random 8:2 split manifest was not retained.",
            "The original multi-agent preprocessing/group manifest was not retained; each raw trajectory is evaluated as a one-agent group.",
            "Training exposure of best_model_indoor.pt is unknown, so TA-GAN results are not claimed as leakage-free held-out performance.",
            "The manuscript's reported 0.079/0.13 TA-GAN value is not treated as a reproduction result here.",
        ],
        "arguments": vars(args),
        "dataset_root": str(dataset_root),
        "dataset_fingerprint_sha256": dataset_fingerprint(records),
        "all_scenes": all_scenes,
        "selected_scenes": selected_scenes,
        "trajectory_files": len(records),
        "parse_failures": len(failures),
        "windows": len(windows),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "environment": environment,
        "methods": method_summary,
        "paired_comparisons": comparisons,
        "motion_regimes": regime_summary,
    }
    with open(str(output_dir / "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    # Keep Windows CP950 consoles from failing on Chinese dataset paths.  The
    # on-disk JSON above remains UTF-8 with unescaped Unicode for readability.
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(PACKAGE_ROOT / "scripts" / "best_model_indoor.pt"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("hash20", "all"), default="hash20")
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
