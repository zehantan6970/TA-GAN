import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "efficiency_e4" / "run_efficiency_benchmark.py"
SPEC = importlib.util.spec_from_file_location("efficiency_e4", str(RUNNER))
E4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E4)


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_cvm_last_extends_last_displacement(self):
        obs = torch.tensor([
            [[0.0, 1.0]],
            [[0.5, 0.75]],
            [[1.25, 0.25]],
        ])
        prediction = E4.cvm_last_prediction(obs, pred_len=3)
        expected = torch.tensor([
            [[2.0, -0.25]],
            [[2.75, -0.75]],
            [[3.5, -1.25]],
        ])
        torch.testing.assert_close(prediction, expected)

    def test_parameter_count_matches_formal_checkpoint_model(self):
        model = E4.E2.load_indoor_generator(
            E4.DEFAULT_CHECKPOINT, "cpu", obs_len=20
        )
        self.assertEqual(E4.count_parameters(model), (4834, 4834))

    def test_latency_summary(self):
        result = E4.summarize_ms([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean_ms"], 2.5)
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["p95_ms"], 4.0)

    def test_batch_one_vectorized_prediction_matches_model_forward(self):
        model = E4.E2.load_indoor_generator(
            E4.DEFAULT_CHECKPOINT, "cpu", obs_len=20
        )
        obs, obs_rel, noise = E4.make_inputs(1, 20, 4, torch.device("cpu"))
        groups = torch.tensor([[0, 1]], dtype=torch.long)
        with torch.no_grad():
            historical = model(obs, obs_rel, groups, noise=noise)
            historical = historical.cumsum(dim=0) + obs[-1].unsqueeze(0)
            vectorized = E4.ta_gan_prediction(model, obs, obs_rel, noise)
        torch.testing.assert_close(historical, vectorized, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
