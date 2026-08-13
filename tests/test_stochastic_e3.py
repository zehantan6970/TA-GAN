"""Tests for the Reviewer 1 stochastic/multimodal E3 runner."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "stochastic_e3" / "run_stochastic_evaluation.py"
SPEC = importlib.util.spec_from_file_location("run_stochastic_evaluation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StochasticMetricTests(unittest.TestCase):
    def test_sample_seed_is_stable_and_separates_indices(self):
        self.assertEqual(MODULE.sample_seed(17, 2), MODULE.sample_seed(17, 2))
        self.assertNotEqual(MODULE.sample_seed(17, 2), MODULE.sample_seed(17, 3))
        self.assertNotEqual(MODULE.sample_seed(17, 2), MODULE.sample_seed(18, 2))

    def test_pairwise_diversity_for_two_candidates(self):
        predictions = np.array([[[[0.0, 0.0], [0.0, 0.0]],
                                 [[1.0, 0.0], [2.0, 0.0]]]])
        endpoint, trajectory = MODULE.pairwise_diversity(predictions)
        self.assertTrue(np.allclose(endpoint, [2.0]))
        self.assertTrue(np.allclose(trajectory, [1.5]))

    def test_pairwise_diversity_zero_for_one_candidate(self):
        predictions = np.zeros((3, 1, 2, 2))
        endpoint, trajectory = MODULE.pairwise_diversity(predictions)
        self.assertTrue(np.array_equal(endpoint, np.zeros(3)))
        self.assertTrue(np.array_equal(trajectory, np.zeros(3)))

    def test_metrics_distinguish_expected_ensemble_and_oracle(self):
        targets = np.zeros((1, 2, 2))
        predictions = np.array([[
            [[1.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [-1.0, 0.0]],
        ]])
        metrics = MODULE.stochastic_metrics(predictions, targets, [1, 2])
        self.assertTrue(np.allclose(metrics["single_ade"], [1.0]))
        self.assertTrue(np.allclose(metrics["expected_ade"], [1.0]))
        self.assertTrue(np.allclose(metrics["ensemble_mean_ade"], [0.0]))
        self.assertTrue(np.allclose(metrics["min_ade"], [1.0]))
        self.assertTrue(np.allclose(metrics["endpoint_diversity"], [2.0]))

    def test_best_of_k_is_monotonic_nonincreasing(self):
        targets = np.zeros((1, 1, 2))
        predictions = np.array([[
            [[3.0, 0.0]], [[2.0, 0.0]], [[1.0, 0.0]],
        ]])
        metrics = MODULE.stochastic_metrics(predictions, targets, [1, 2, 3])
        values = [metrics["best_of_k_ade"][k][0] for k in (1, 2, 3)]
        self.assertEqual(values, [3.0, 2.0, 1.0])

    def test_invalid_prediction_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.pairwise_diversity(np.zeros((2, 3, 4)))

    def test_vectorized_one_agent_forward_matches_historical_group_loop(self):
        model = MODULE.E2.Trajectory_Generator(
            obs_len=20,
            embedding_dim=16,
            encoder_input_dim=16,
            encoder_output_dim=16,
            encoder_mlp_dim=16,
            encoder_num_head=2,
            drop_rate=0,
            rel_traj_dim=16,
            noise_dim=4,
            merge_mlp_dim=16,
        ).eval()
        torch.manual_seed(9)
        obs = torch.randn(20, 5, 2)
        obs_rel = torch.zeros_like(obs)
        obs_rel[1:] = obs[1:] - obs[:-1]
        noise = torch.randn(5, 20, 4)
        boundaries = torch.arange(6, dtype=torch.long)
        groups = torch.stack([boundaries[:-1], boundaries[1:]], dim=1)
        with torch.no_grad():
            historical = model(obs, obs_rel, groups, noise=noise)
            vectorized = MODULE.forward_independent_one_agent_groups(
                model, obs, obs_rel, noise
            )
        torch.testing.assert_close(
            historical, vectorized, rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
