"""Tests for the Reviewer 1.3 constant-velocity comparison runner."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY_ROOT / "experiments" / "cvm_e2" / "run_cvm_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("run_cvm_comparison", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstantVelocityTests(unittest.TestCase):
    def test_both_variants_are_exact_for_constant_velocity_irregular_times(self):
        times = np.cumsum(np.linspace(0.08, 0.12, 40))
        velocity = np.array([0.4, -0.2])
        positions = np.array([1.0, 2.0]) + np.outer(times, velocity)
        last = MODULE.cvm_last_two(times[:20], positions[:20], times[20:])
        least_squares = MODULE.cvm_least_squares(
            times[:20], positions[:20], times[20:]
        )
        self.assertTrue(np.allclose(last, positions[20:], atol=1e-12))
        self.assertTrue(np.allclose(least_squares, positions[20:], atol=1e-12))

    def test_last_two_rejects_nonincreasing_time(self):
        with self.assertRaisesRegex(ValueError, "must increase"):
            MODULE.cvm_last_two(
                np.array([0.0, 0.0]), np.zeros((2, 2)), np.array([0.1])
            )


class DatasetProtocolTests(unittest.TestCase):
    def test_reader_and_window_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "scene1" / "agent"
            scene.mkdir(parents=True)
            path = scene / "trajectory.txt"
            timestamps = 1_000_000 + np.arange(60) * 100_000
            positions = np.column_stack(
                [np.arange(60) * 0.01, np.arange(60) * -0.02]
            )
            np.savetxt(str(path), np.column_stack([timestamps, positions]))

            record = MODULE.read_trajectory(path, root)
            windows = MODULE.make_windows(record, 20, 20, 10)

            self.assertEqual(record.scene, "scene1")
            self.assertEqual(len(windows), 3)
            self.assertEqual(windows[1].start_index, 10)
            self.assertAlmostEqual(windows[0].timestamps_s[-1], 3.9)

    def test_hash20_assignment_is_stable(self):
        first = [
            scene
            for scene in ("scene1", "scene2", "scene3", "scene4")
            if MODULE.scene_is_in_hash20(scene)
        ]
        second = [
            scene
            for scene in ("scene1", "scene2", "scene3", "scene4")
            if MODULE.scene_is_in_hash20(scene)
        ]
        self.assertEqual(first, second)

    def test_cluster_bootstrap_is_deterministic(self):
        values = np.array([1.0, 2.0, 4.0, 5.0])
        clusters = np.array(["a", "a", "b", "b"])
        first = MODULE.cluster_bootstrap(values, clusters, 100, 17)
        second = MODULE.cluster_bootstrap(values, clusters, 100, 17)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

