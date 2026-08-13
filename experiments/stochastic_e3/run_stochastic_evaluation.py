"""Reviewer 1 enrichment E3: stochastic TA-GAN evaluation.

This runner deliberately reuses the documented E2 retained-data protocol so
that the CVM and stochastic reports refer to the same trajectory files and
windows. It evaluates accuracy, oracle best-of-K behavior, diversity that does
not use ground truth, and variation across independent base seeds.
"""

from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
E2_RUNNER = REPOSITORY_ROOT / "experiments" / "cvm_e2" / "run_cvm_comparison.py"
PACKAGE_ROOT = REPOSITORY_ROOT / "ta_gan"


def load_e2_module():
    spec = importlib.util.spec_from_file_location("cvm_e2_protocol", str(E2_RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E2 = load_e2_module()


def parse_base_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one base seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("base seeds must be unique")
    return seeds


def sample_seed(base_seed, sample_index):
    """Derive non-overlapping, reproducible uint32 seeds."""
    sequence = np.random.SeedSequence([int(base_seed), int(sample_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def pairwise_diversity(predictions):
    """Return ground-truth-independent pairwise endpoint/trajectory diversity.

    Args:
        predictions: ``[windows, samples, future_steps, xy]`` array.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError("predictions must have shape [N, K, T, 2]")
    samples = predictions.shape[1]
    if samples < 2:
        zeros = np.zeros(predictions.shape[0], dtype=np.float64)
        return zeros.copy(), zeros.copy()
    endpoint_sum = np.zeros(predictions.shape[0], dtype=np.float64)
    trajectory_sum = np.zeros(predictions.shape[0], dtype=np.float64)
    pairs = 0
    # Accumulate one candidate pair at a time. Materializing [N,K,K,T,2]
    # would create a gigabyte-scale temporary array for the complete protocol.
    for first in range(samples - 1):
        for second in range(first + 1, samples):
            delta = predictions[:, first] - predictions[:, second]
            endpoint_sum += np.linalg.norm(delta[:, -1], axis=-1)
            trajectory_sum += np.linalg.norm(delta, axis=-1).mean(axis=-1)
            pairs += 1
    return endpoint_sum / pairs, trajectory_sum / pairs


def stochastic_metrics(predictions, targets, best_of_k_values):
    """Compute per-window accuracy and diversity for one base seed."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    distances = np.linalg.norm(predictions - targets[:, None, :, :], axis=-1)
    ade = distances.mean(axis=-1)
    fde = distances[:, :, -1]
    mean_prediction = predictions.mean(axis=1)
    mean_distances = np.linalg.norm(mean_prediction - targets, axis=-1)
    endpoint_diversity, trajectory_diversity = pairwise_diversity(predictions)
    result = {
        "ade_samples": ade,
        "fde_samples": fde,
        "single_ade": ade[:, 0],
        "single_fde": fde[:, 0],
        "expected_ade": ade.mean(axis=1),
        "expected_fde": fde.mean(axis=1),
        "ensemble_mean_ade": mean_distances.mean(axis=1),
        "ensemble_mean_fde": mean_distances[:, -1],
        "min_ade": ade.min(axis=1),
        "min_fde": fde.min(axis=1),
        "endpoint_diversity": endpoint_diversity,
        "trajectory_diversity": trajectory_diversity,
        "best_of_k_ade": {},
        "best_of_k_fde": {},
    }
    for value in best_of_k_values:
        result["best_of_k_ade"][value] = ade[:, :value].min(axis=1)
        result["best_of_k_fde"][value] = fde[:, :value].min(axis=1)
    return result


def forward_independent_one_agent_groups(model, obs, obs_rel, noise):
    """Vectorized equivalent of the generator for independent one-agent groups.

    The reconstructed protocol defines every window as a separate one-agent
    group. In that special case each pairwise relative position is zero and the
    historical group loop can be evaluated for every window in one tensor.
    """
    embedding = model.traj_embedding(obs_rel).transpose(0, 1)
    encoder_output = model.trans_encoder(embedding)
    relative_positions = torch.zeros(
        encoder_output.size(0), model.obs_len, 2,
        device=obs.device, dtype=obs.dtype
    )
    relative_embedding = model.rel_embedding(relative_positions)
    merged = model.merge_mlp(
        torch.cat((encoder_output, relative_embedding), dim=2)
    )
    weights = model.sigmoid(
        model.social_mlp(torch.flatten(merged, start_dim=1))
    )
    social_features = weights.unsqueeze(1) * merged
    decoder_input = model.add_noise(social_features, noise=noise)
    return model.trans_decoder(decoder_input).transpose(0, 1)


def predict_samples(model, windows, obs_len, pred_len, samples, batch_size,
                    base_seed, device):
    count = len(windows)
    predictions = np.empty((count, samples, pred_len, 2), dtype=np.float32)

    def run_batch(batch_windows, noise_array):
        obs_np = np.stack(
            [item.positions[:obs_len] for item in batch_windows], axis=1
        ).astype(np.float32)
        obs = torch.from_numpy(obs_np).to(device)
        obs_rel = torch.zeros_like(obs)
        obs_rel[1:] = obs[1:] - obs[:-1]
        noise = torch.from_numpy(noise_array).to(device)
        with torch.no_grad():
            pred_rel = forward_independent_one_agent_groups(
                model, obs, obs_rel, noise
            )
            pred_abs = pred_rel.cumsum(dim=0) + obs[-1].unsqueeze(0)
        return pred_abs.transpose(0, 1).cpu().numpy()

    for index in range(samples):
        random = np.random.RandomState(sample_seed(base_seed, index))
        noise = random.standard_normal((count, obs_len, 4)).astype(np.float32)
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            predictions[start:end, index] = run_batch(
                windows[start:end], noise[start:end]
            )
    return predictions


def summarize(values, clusters, iterations, seed):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std_across_windows": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "ci95_source_file_cluster_bootstrap": E2.cluster_bootstrap(
            values, clusters, iterations, seed
        ),
    }


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_examples(rows):
    """Select examples by fixed metric rules rather than visual preference."""
    endpoint = np.asarray([row["endpoint_diversity_m"] for row in rows])
    single = np.asarray([row["single_ade_m"] for row in rows])
    oracle_gain = single - np.asarray([row["minade_20_m"] for row in rows])
    chosen = [
        ("median diversity", int(np.argmin(np.abs(endpoint - np.median(endpoint))))),
        ("highest diversity", int(np.argmax(endpoint))),
        ("largest oracle gain", int(np.argmax(oracle_gain))),
        ("smallest oracle gain", int(np.argmin(oracle_gain))),
    ]
    unique = []
    used = set()
    for label, index in chosen:
        if index not in used:
            unique.append((label, index))
            used.add(index)
    return unique


def plot_best_of_k(path, rows):
    k_values = [row["k"] for row in rows]
    ade = [row["minade_m"] for row in rows]
    fde = [row["minfde_m"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.6, 4.3))
    axis.plot(k_values, ade, "o-", color="#226f73", label="minADE@K")
    axis.plot(k_values, fde, "s-", color="#a33d35", label="minFDE@K")
    axis.set_xlabel("Number of stochastic candidates K")
    axis.set_ylabel("Displacement error (m)")
    axis.set_xticks(k_values)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def plot_diversity(path, rows):
    endpoint = np.asarray([row["endpoint_diversity_m"] for row in rows])
    trajectory = np.asarray([row["trajectory_diversity_m"] for row in rows])
    gain = np.asarray([row["single_ade_m"] - row["minade_20_m"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(endpoint, bins=50, color="#226f73", alpha=0.85)
    axes[0].axvline(np.median(endpoint), color="#202020", linestyle="--")
    axes[0].set_xlabel("Mean pairwise endpoint distance (m)")
    axes[0].set_ylabel("Windows")
    axes[0].set_title("Endpoint diversity")
    axes[1].scatter(trajectory, gain, s=7, alpha=0.20, color="#a33d35")
    axes[1].set_xlabel("Mean pairwise trajectory distance (m)")
    axes[1].set_ylabel("Single ADE - minADE@20 (m)")
    axes[1].set_title("Diversity and oracle gain")
    figure.tight_layout()
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def plot_examples(path, examples, windows, predictions, rows, obs_len):
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    for axis, (label, index) in zip(axes, examples):
        positions = windows[index].positions
        axis.plot(positions[:obs_len, 0], positions[:obs_len, 1], "o-", ms=2,
                  color="#36496b", label="Observed")
        axis.plot(positions[obs_len:, 0], positions[obs_len:, 1], "o-", ms=2,
                  color="#202020", label="Ground truth")
        for sample in predictions[index]:
            axis.plot(sample[:, 0], sample[:, 1], color="#2b7a78", alpha=0.20,
                      linewidth=1)
        mean_prediction = predictions[index].mean(axis=0)
        axis.plot(mean_prediction[:, 0], mean_prediction[:, 1], "--",
                  color="#b23a31", linewidth=1.5, label="Candidate mean")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_title(
            "{}\nendpoint diversity {:.3f} m, minADE@20 {:.3f} m".format(
                label, rows[index]["endpoint_diversity_m"],
                rows[index]["minade_20_m"]
            ), fontsize=9
        )
    for axis in axes[len(examples):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=[0, 0.06, 1, 1])
    figure.savefig(str(path), dpi=180)
    plt.close(figure)


def run(args):
    dataset_root = Path(args.dataset_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seeds = args.base_seeds
    best_of_k_values = sorted(set(args.best_of_k))
    if best_of_k_values[-1] > args.samples or best_of_k_values[0] < 1:
        raise ValueError("best-of-K values must be between 1 and --samples")

    records, failures = E2.discover_records(
        dataset_root, scope=args.scope, max_files=args.max_files
    )
    windows = []
    for record in records:
        windows.extend(E2.make_windows(record, args.obs_len, args.pred_len, args.stride))
    if not windows:
        raise RuntimeError("no valid windows found")
    targets = np.stack([window.positions[args.obs_len:] for window in windows])
    clusters = np.asarray([window.source_file for window in windows])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E2.load_indoor_generator(checkpoint, device, args.obs_len)

    seed_results = []
    presentation_predictions = None
    metric_names = [
        "single_ade", "single_fde", "expected_ade", "expected_fde",
        "ensemble_mean_ade", "ensemble_mean_fde", "min_ade", "min_fde",
        "endpoint_diversity", "trajectory_diversity",
    ]
    metric_stack = {name: [] for name in metric_names}
    all_ade_samples = []
    all_fde_samples = []
    best_stack = {
        (metric, k): [] for metric in ("ade", "fde") for k in best_of_k_values
    }

    for seed_index, base_seed in enumerate(base_seeds):
        predictions = predict_samples(
            model, windows, args.obs_len, args.pred_len, args.samples,
            args.batch_size, base_seed, device
        )
        metrics = stochastic_metrics(predictions, targets, best_of_k_values)
        if seed_index == 0:
            presentation_predictions = predictions.copy()
        all_ade_samples.append(metrics["ade_samples"])
        all_fde_samples.append(metrics["fde_samples"])
        for name in metric_names:
            metric_stack[name].append(metrics[name])
        for k in best_of_k_values:
            best_stack[("ade", k)].append(metrics["best_of_k_ade"][k])
            best_stack[("fde", k)].append(metrics["best_of_k_fde"][k])
        seed_results.append({
            "base_seed": base_seed,
            "single_ade_m": float(metrics["single_ade"].mean()),
            "single_fde_m": float(metrics["single_fde"].mean()),
            "expected_ade_m": float(metrics["expected_ade"].mean()),
            "expected_fde_m": float(metrics["expected_fde"].mean()),
            "ensemble_mean_ade_m": float(metrics["ensemble_mean_ade"].mean()),
            "ensemble_mean_fde_m": float(metrics["ensemble_mean_fde"].mean()),
            "minade_20_m": float(metrics["min_ade"].mean()),
            "minfde_20_m": float(metrics["min_fde"].mean()),
            "endpoint_diversity_m": float(metrics["endpoint_diversity"].mean()),
            "trajectory_diversity_m": float(metrics["trajectory_diversity"].mean()),
        })

    averaged = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in metric_stack.items()
    }
    per_window_rows = []
    for index, window in enumerate(windows):
        regime, heading, path_length = E2.motion_regime(window.positions, args.obs_len)
        per_window_rows.append({
            "sample_id": window.sample_id,
            "source_file": window.source_file,
            "scene": window.scene,
            "start_index": window.start_index,
            "regime": regime,
            "heading_change_deg": heading,
            "path_length_m": path_length,
            "single_ade_m": averaged["single_ade"][index],
            "single_fde_m": averaged["single_fde"][index],
            "expected_ade_m": averaged["expected_ade"][index],
            "expected_fde_m": averaged["expected_fde"][index],
            "ensemble_mean_ade_m": averaged["ensemble_mean_ade"][index],
            "ensemble_mean_fde_m": averaged["ensemble_mean_fde"][index],
            "minade_20_m": averaged["min_ade"][index],
            "minfde_20_m": averaged["min_fde"][index],
            "endpoint_diversity_m": averaged["endpoint_diversity"][index],
            "trajectory_diversity_m": averaged["trajectory_diversity"][index],
            "single_ade_seed_std_m": np.std(
                np.stack(metric_stack["single_ade"], axis=0)[:, index], ddof=0
            ),
        })

    summary_metrics = {}
    bootstrap_seed = base_seeds[0]
    for name in metric_names:
        summary_metrics[name] = summarize(
            averaged[name], clusters, args.bootstrap_iterations, bootstrap_seed
        )
    summary_metrics["endpoint_diversity"]["fraction_below_0.01_m"] = float(
        np.mean(averaged["endpoint_diversity"] < 0.01)
    )
    summary_metrics["endpoint_diversity"]["fraction_below_0.05_m"] = float(
        np.mean(averaged["endpoint_diversity"] < 0.05)
    )

    best_rows = []
    for k in best_of_k_values:
        ade_values = np.mean(np.stack(best_stack[("ade", k)], axis=0), axis=0)
        fde_values = np.mean(np.stack(best_stack[("fde", k)], axis=0), axis=0)
        best_rows.append({
            "k": k,
            "minade_m": float(ade_values.mean()),
            "minfde_m": float(fde_values.mean()),
            "minade_ci95_low_m": E2.cluster_bootstrap(
                ade_values, clusters, args.bootstrap_iterations, bootstrap_seed
            )[0],
            "minade_ci95_high_m": E2.cluster_bootstrap(
                ade_values, clusters, args.bootstrap_iterations, bootstrap_seed
            )[1],
            "minfde_ci95_low_m": E2.cluster_bootstrap(
                fde_values, clusters, args.bootstrap_iterations, bootstrap_seed
            )[0],
            "minfde_ci95_high_m": E2.cluster_bootstrap(
                fde_values, clusters, args.bootstrap_iterations, bootstrap_seed
            )[1],
        })

    regimes = {}
    for regime in sorted(set(row["regime"] for row in per_window_rows)):
        selected = [row for row in per_window_rows if row["regime"] == regime]
        regimes[regime] = {
            "windows": len(selected),
            "single_ade_m": float(np.mean([row["single_ade_m"] for row in selected])),
            "expected_ade_m": float(np.mean([row["expected_ade_m"] for row in selected])),
            "minade_20_m": float(np.mean([row["minade_20_m"] for row in selected])),
            "endpoint_diversity_m": float(np.mean(
                [row["endpoint_diversity_m"] for row in selected]
            )),
        }

    manifest_rows = [{
        "relative_path": record.relative_path,
        "scene": record.scene,
        "points": len(record.timestamps_us),
        "windows": max(0, (len(record.timestamps_us) - args.obs_len - args.pred_len)
                       // args.stride + 1),
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    } for record in records]
    write_csv(output_dir / "dataset_manifest.csv", manifest_rows)
    write_csv(output_dir / "per_window_metrics.csv", per_window_rows)
    write_csv(output_dir / "per_seed_summary.csv", seed_results)
    write_csv(output_dir / "best_of_k.csv", best_rows)
    if failures:
        write_csv(output_dir / "parse_failures.csv", failures)
    np.savez_compressed(
        str(output_dir / "per_sample_metrics.npz"),
        ade_m=np.stack(all_ade_samples, axis=0),
        fde_m=np.stack(all_fde_samples, axis=0),
        base_seeds=np.asarray(base_seeds, dtype=np.int64),
        sample_seeds=np.asarray(
            [[sample_seed(seed, index) for index in range(args.samples)]
             for seed in base_seeds], dtype=np.uint32
        ),
    )

    examples = select_examples(per_window_rows)
    example_rows = [{
        "selection_rule": label,
        "window_index": index,
        "sample_id": per_window_rows[index]["sample_id"],
        "source_file": per_window_rows[index]["source_file"],
    } for label, index in examples]
    write_csv(output_dir / "qualitative_manifest.csv", example_rows)
    selected_indices = np.asarray([index for _, index in examples], dtype=np.int64)
    np.savez_compressed(
        str(output_dir / "qualitative_predictions.npz"),
        window_indices=selected_indices,
        observed=np.stack([windows[index].positions[:args.obs_len]
                           for index in selected_indices]),
        targets=np.stack([windows[index].positions[args.obs_len:]
                          for index in selected_indices]),
        predictions=presentation_predictions[selected_indices],
        presentation_base_seed=np.asarray([base_seeds[0]], dtype=np.int64),
    )
    plot_best_of_k(output_dir / "best_of_k_curve.png", best_rows)
    plot_diversity(output_dir / "diversity_analysis.png", per_window_rows)
    plot_examples(output_dir / "stochastic_examples.png", examples, windows,
                  presentation_predictions, per_window_rows, args.obs_len)

    selected_scenes = sorted(set(record.scene for record in records))
    seed_metric_keys = [key for key in seed_results[0] if key != "base_seed"]
    seed_stability = {
        key: {
            "mean": float(np.mean([row[key] for row in seed_results])),
            "std_across_base_seeds": float(np.std(
                [row[key] for row in seed_results], ddof=0
            )),
            "min": float(np.min([row[key] for row in seed_results])),
            "max": float(np.max([row[key] for row in seed_results])),
        } for key in seed_metric_keys
    }
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_status": "RECONSTRUCTED_RETAINED_DATA_PROTOCOL",
        "limitations": [
            "The original random 8:2 split manifest was not retained.",
            "The original multi-agent group manifest was not retained; each trajectory file is evaluated as a one-agent group.",
            "Checkpoint training exposure is unknown, so results are not claimed as leakage-free held-out performance.",
            "minADE/minFDE@K are oracle metrics that use ground truth for sample selection and are not deployable selection rules.",
            "Diversity measures spread, not correctness or semantic multimodality.",
        ],
        "arguments": {**vars(args), "base_seeds": base_seeds},
        "selected_scenes": selected_scenes,
        "trajectory_files": len(records),
        "windows": len(windows),
        "dataset_fingerprint_sha256": E2.dataset_fingerprint(records),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": E2.sha256_file(checkpoint),
        "runner_sha256": E2.sha256_file(Path(__file__).resolve()),
        "shared_e2_protocol_runner_sha256": E2.sha256_file(E2_RUNNER),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
        },
        "metrics_averaged_across_base_seeds": summary_metrics,
        "seed_stability": seed_stability,
        "best_of_k": best_rows,
        "motion_regimes": regimes,
    }
    with open(str(output_dir / "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    artifact_rows = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "artifact_sha256.csv":
            artifact_rows.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": E2.sha256_file(path),
            })
    write_csv(
        output_dir / "artifact_sha256.csv",
        artifact_rows,
        ["filename", "size_bytes", "sha256"],
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--checkpoint", default=str(PACKAGE_ROOT / "scripts" / "best_model_indoor.pt")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("hash20", "all"), default="hash20")
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--pred-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--base-seeds", type=parse_base_seeds,
        default=parse_base_seeds("20260812,20261812,20262812,20263812,20264812")
    )
    parser.add_argument("--best-of-k", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
