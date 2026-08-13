"""Regression tests for the recovered TA-GAN Transformer model."""

import os
import sys
import unittest

import torch


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPOSITORY_ROOT, "ta_gan")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from sgan.models_transformer import (  # noqa: E402
    Trajectory_Discriminator,
    Trajectory_Generator,
)
from sgan.models_transformer_ori import (  # noqa: E402
    Trajectory_Generator as IndoorTrajectoryGenerator,
)


def build_generator(obs_len=8):
    return Trajectory_Generator(
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


def sample_batch(obs_len=8, device="cpu"):
    torch.manual_seed(11)
    obs_traj = torch.randn(obs_len, 3, 2, device=device)
    obs_traj_rel = torch.zeros_like(obs_traj)
    obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
    seq_start_end = torch.tensor([[0, 2], [2, 3]], device=device)
    return obs_traj, obs_traj_rel, seq_start_end


class RecoveredGeneratorTests(unittest.TestCase):
    def test_cpu_forward_shape_and_finite_values(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()
        noise = torch.zeros(3, 8, 4)

        with torch.no_grad():
            output = model(obs_traj, obs_traj_rel, seq_start_end, noise=noise)

        self.assertEqual(tuple(output.shape), (8, 3, 2))
        self.assertFalse(output.is_cuda)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_explicit_noise_is_reproducible_and_changes_predictions(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()
        zero_noise = torch.zeros(3, 8, 4)
        one_noise = torch.ones(3, 8, 4)

        with torch.no_grad():
            first = model(obs_traj, obs_traj_rel, seq_start_end, noise=zero_noise)
            repeated = model(
                obs_traj, obs_traj_rel, seq_start_end, noise=zero_noise
            )
            changed = model(obs_traj, obs_traj_rel, seq_start_end, noise=one_noise)

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.allclose(first, changed))

    def test_invalid_group_partition_is_rejected(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, _ = sample_batch()
        invalid_groups = torch.tensor([[0, 1], [2, 3]])

        with self.assertRaisesRegex(ValueError, "contiguous"):
            model(obs_traj, obs_traj_rel, invalid_groups)

    def test_invalid_noise_shape_is_rejected(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()

        with self.assertRaisesRegex(ValueError, "noise must have shape"):
            model(
                obs_traj,
                obs_traj_rel,
                seq_start_end,
                noise=torch.zeros(3, 8, 3),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward(self):
        model = build_generator().cuda().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch(device="cuda")

        with torch.no_grad():
            output = model(obs_traj, obs_traj_rel, seq_start_end)

        self.assertTrue(output.is_cuda)
        self.assertEqual(tuple(output.shape), (8, 3, 2))
        self.assertTrue(torch.isfinite(output).all().item())


class RecoveredDiscriminatorTests(unittest.TestCase):
    def test_discriminator_forward_shape(self):
        discriminator = Trajectory_Discriminator(
            obs_len=16,
            embedding_dim=16,
            encoder_input_dim=16,
            encoder_output_dim=16,
            mlp_hid_dim=16,
            num_head=2,
            drop_rate=0,
        ).eval()
        trajectory = torch.randn(16, 3, 2)
        seq_start_end = torch.tensor([[0, 2], [2, 3]])

        with torch.no_grad():
            scores = discriminator(trajectory, seq_start_end)

        self.assertEqual(tuple(scores.shape), (3, 1))
        self.assertTrue(((scores >= 0) & (scores <= 1)).all().item())


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_indoor_checkpoint_strictly_matches_ori_architecture(self):
        model = IndoorTrajectoryGenerator(
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
        )
        checkpoint_path = os.path.join(
            PACKAGE_ROOT, "scripts", "best_model_indoor.pt"
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        result = model.load_state_dict(state_dict, strict=True)

        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4834)

    def test_recovered_encoder_is_a_distinct_checkpoint_schema(self):
        recovered_keys = set(build_generator(obs_len=20).state_dict().keys())
        checkpoint_path = os.path.join(
            PACKAGE_ROOT, "scripts", "best_model_indoor.pt"
        )
        indoor_keys = set(torch.load(checkpoint_path, map_location="cpu").keys())

        self.assertIn("trans_encoder.blocks.0.q.weight", recovered_keys)
        self.assertIn("trans_encoder.q.weight", indoor_keys)
        self.assertNotEqual(recovered_keys, indoor_keys)

    def test_indoor_checkpoint_cpu_forward_with_explicit_noise(self):
        model = IndoorTrajectoryGenerator(
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
        checkpoint_path = os.path.join(
            PACKAGE_ROOT, "scripts", "best_model_indoor.pt"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        obs_traj, obs_traj_rel, seq_start_end = sample_batch(obs_len=20)
        noise = torch.zeros(3, 20, 4)

        with torch.no_grad():
            first = model(
                obs_traj, obs_traj_rel, seq_start_end, noise=noise
            )
            repeated = model(
                obs_traj, obs_traj_rel, seq_start_end, noise=noise
            )

        self.assertEqual(tuple(first.shape), (20, 3, 2))
        self.assertTrue(torch.equal(first, repeated))

    def test_indoor_checkpoint_rejects_invalid_noise_shape(self):
        model = IndoorTrajectoryGenerator(
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
        obs_traj, obs_traj_rel, seq_start_end = sample_batch(obs_len=20)

        with self.assertRaisesRegex(ValueError, "noise must have shape"):
            model(
                obs_traj,
                obs_traj_rel,
                seq_start_end,
                noise=torch.zeros(3, 20, 3),
            )


if __name__ == "__main__":
    unittest.main()
