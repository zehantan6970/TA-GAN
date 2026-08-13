"""Regression tests for the restored Social-GAN baseline module."""

import importlib.util
import os
import sys
import unittest

import torch


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPOSITORY_ROOT, "ta_gan")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from sgan.models import (  # noqa: E402
    PoolHiddenNet,
    SocialPooling,
    TrajectoryDiscriminator,
    TrajectoryGenerator,
)


def sample_batch(device="cpu"):
    torch.manual_seed(19)
    obs_traj = torch.randn(4, 3, 2, device=device)
    obs_traj_rel = torch.zeros_like(obs_traj)
    obs_traj_rel[1:] = obs_traj[1:] - obs_traj[:-1]
    seq_start_end = torch.tensor([[0, 2], [2, 3]], device=device)
    return obs_traj, obs_traj_rel, seq_start_end


def build_generator(noise_mix_type="ped"):
    return TrajectoryGenerator(
        obs_len=4,
        pred_len=3,
        embedding_dim=8,
        encoder_h_dim=8,
        decoder_h_dim=12,
        mlp_dim=16,
        noise_dim=(4,),
        noise_mix_type=noise_mix_type,
        pooling_type="pool_net",
        pool_every_timestep=True,
        bottleneck_dim=8,
        batch_norm=False,
    )


class GeneratorTests(unittest.TestCase):
    def test_cpu_forward_with_explicit_pedestrian_noise(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()
        noise = torch.zeros(3, 4)

        with torch.no_grad():
            output = model(
                obs_traj,
                obs_traj_rel,
                seq_start_end,
                user_noise=noise,
            )

        self.assertEqual(tuple(output.shape), (3, 3, 2))
        self.assertFalse(output.is_cuda)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_global_noise_is_reproducible(self):
        model = build_generator(noise_mix_type="global").eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()
        noise = torch.tensor([[0.25] * 4, [-0.5] * 4])

        with torch.no_grad():
            first = model(
                obs_traj, obs_traj_rel, seq_start_end, user_noise=noise
            )
            repeated = model(
                obs_traj, obs_traj_rel, seq_start_end, user_noise=noise
            )

        self.assertTrue(torch.equal(first, repeated))

    def test_invalid_user_noise_shape_is_rejected(self):
        model = build_generator().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch()

        with self.assertRaisesRegex(ValueError, "user_noise must have shape"):
            model(
                obs_traj,
                obs_traj_rel,
                seq_start_end,
                user_noise=torch.zeros(2, 4),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_uses_input_device(self):
        model = build_generator().cuda().eval()
        obs_traj, obs_traj_rel, seq_start_end = sample_batch(device="cuda")

        with torch.no_grad():
            output = model(obs_traj, obs_traj_rel, seq_start_end)

        self.assertTrue(output.is_cuda)
        self.assertEqual(tuple(output.shape), (3, 3, 2))


class PoolingTests(unittest.TestCase):
    def test_pairwise_pooling_shape(self):
        layer = PoolHiddenNet(
            embedding_dim=4,
            h_dim=8,
            mlp_dim=16,
            bottleneck_dim=6,
            batch_norm=False,
        ).eval()
        _, _, seq_start_end = sample_batch()
        result = layer(torch.randn(1, 3, 8), seq_start_end, torch.randn(3, 2))
        self.assertEqual(tuple(result.shape), (3, 6))

    def test_social_grid_pooling_shape(self):
        layer = SocialPooling(
            h_dim=8, grid_size=4, pool_dim=6, batch_norm=False
        ).eval()
        _, _, seq_start_end = sample_batch()
        result = layer(torch.randn(1, 3, 8), seq_start_end, torch.randn(3, 2))
        self.assertEqual(tuple(result.shape), (3, 6))


class DiscriminatorTests(unittest.TestCase):
    def _run_discriminator(self, discriminator_type):
        model = TrajectoryDiscriminator(
            obs_len=4,
            pred_len=3,
            embedding_dim=8,
            h_dim=8,
            mlp_dim=16,
            batch_norm=False,
            d_type=discriminator_type,
        ).eval()
        trajectory = torch.randn(7, 3, 2)
        relative = torch.zeros_like(trajectory)
        relative[1:] = trajectory[1:] - trajectory[:-1]
        seq_start_end = torch.tensor([[0, 2], [2, 3]])
        with torch.no_grad():
            return model(trajectory, relative, seq_start_end)

    def test_local_discriminator(self):
        self.assertEqual(tuple(self._run_discriminator("local").shape), (3, 1))

    def test_global_discriminator(self):
        self.assertEqual(tuple(self._run_discriminator("global").shape), (3, 1))


class RecoveryCompatibilityTests(unittest.TestCase):
    def test_parameter_schema_and_initialization_match_upstream(self):
        upstream_path = os.path.join(
            REPOSITORY_ROOT, "upstream", "social-gan", "sgan", "models.py"
        )
        spec = importlib.util.spec_from_file_location(
            "social_gan_upstream_models", upstream_path
        )
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)

        kwargs = dict(
            obs_len=4,
            pred_len=3,
            embedding_dim=8,
            encoder_h_dim=8,
            decoder_h_dim=12,
            mlp_dim=16,
            noise_dim=(4,),
            pooling_type="pool_net",
            bottleneck_dim=8,
            batch_norm=False,
        )
        torch.manual_seed(73)
        restored = TrajectoryGenerator(**kwargs)
        torch.manual_seed(73)
        original = upstream.TrajectoryGenerator(**kwargs)

        restored_state = restored.state_dict()
        original_state = original.state_dict()
        self.assertEqual(list(restored_state), list(original_state))
        for name in restored_state:
            self.assertTrue(
                torch.equal(restored_state[name], original_state[name]), name
            )

    def test_evaluate_model_import_resolves_restored_module(self):
        import scripts.evaluate_model as evaluate_model

        self.assertIs(evaluate_model.TrajectoryGenerator, TrajectoryGenerator)


if __name__ == "__main__":
    unittest.main()
