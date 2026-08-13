"""Transformer-GAN trajectory model recovered from the retained bytecode.

The repository originally shipped only
``__pycache__/models_transformer.cpython-36.pyc``. The architecture below was
recovered from that Python 3.6 bytecode and cross-checked against the training
scripts. It is intentionally separate from ``models_transformer_ori.py``:

* this module uses a two-block residual Transformer encoder and group-scale
  normalization;
* ``models_transformer_ori.py`` uses a single attention encoder and is the
  architecture matched by ``scripts/best_model_indoor.pt``.

The recovery preserves parameter names and the historical three-argument
generator API. Device-safe noise generation and an optional explicit noise
tensor were added so that CPU tests and reproducible stochastic evaluation are
possible without changing existing callers.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class Traj_embedding(nn.Module):
    """Embed each 2D trajectory point and apply feature normalization."""

    def __init__(self, embedding_dim=64):
        super(Traj_embedding, self).__init__()
        self.embbedding_dim = embedding_dim
        self.tra_embedding = nn.Linear(2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, obs_traj):
        if obs_traj.dim() != 3 or obs_traj.size(-1) != 2:
            raise ValueError("trajectory input must have shape [time, agents, 2]")

        agent_count = obs_traj.size(1)
        embedded = self.tra_embedding(obs_traj.contiguous().view(-1, 2))
        embedded = embedded.view(-1, agent_count, self.embbedding_dim)
        return self.norm(embedded)


class Mlp(nn.Module):
    """Two-layer feed-forward block used throughout the model."""

    def __init__(self, input_dim, hidden_dim, out_dim, drop_rate=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.act_layer = nn.LeakyReLU()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention_Block(nn.Module):
    """Residual temporal self-attention block for one agent trajectory."""

    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_hid_dim,
        num_head=1,
        obs_len=16,
        drop_rate=0.0,
    ):
        super(Attention_Block, self).__init__()
        if input_dim != output_dim:
            raise ValueError("input_dim must equal output_dim for residual attention")
        if output_dim % num_head != 0:
            raise ValueError("output_dim must be divisible by num_head")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.obs_len = obs_len
        self.q = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.kv = nn.Linear(self.input_dim, self.output_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(p=drop_rate)
        self.proj = nn.Linear(output_dim, output_dim)
        self.proj_drop = nn.Dropout(p=drop_rate)
        self.mlp = Mlp(output_dim, mlp_hid_dim, output_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(output_dim)

    def forward(self, obs_traj_embedding):
        residual = obs_traj_embedding
        normalized = self.norm1(obs_traj_embedding)
        batch_size, sequence_length, feature_dim = normalized.shape

        q = self.q(normalized).reshape(
            batch_size,
            sequence_length,
            self.num_head,
            feature_dim // self.num_head,
        ).permute(0, 2, 1, 3)
        k_v = self.kv(normalized).reshape(
            batch_size,
            sequence_length,
            2,
            self.num_head,
            feature_dim // self.num_head,
        ).permute(2, 0, 3, 1, 4)
        k, v = k_v[0], k_v[1]

        attention = q @ k.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(
            batch_size, sequence_length, feature_dim
        )
        x = self.proj_drop(self.proj(x))
        x = x + residual
        return x + self.mlp(self.norm2(x))


class Transformer_Encoder(nn.Module):
    """Add learned positions and apply stacked temporal attention blocks."""

    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_hid_dim,
        num_head=1,
        obs_len=16,
        drop_rate=0.0,
        num_block=2,
    ):
        super(Transformer_Encoder, self).__init__()
        if num_block < 1:
            raise ValueError("num_block must be positive")

        self.blocks = nn.ModuleList(
            [
                Attention_Block(
                    input_dim,
                    output_dim,
                    mlp_hid_dim,
                    num_head,
                    obs_len,
                    drop_rate,
                )
                for _ in range(num_block)
            ]
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_len, input_dim))
        trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, obs_traj_embedding):
        if obs_traj_embedding.size(1) != self.pos_emb.size(1):
            raise ValueError("input sequence length does not match configured obs_len")

        x = obs_traj_embedding + self.pos_emb
        for block in self.blocks:
            x = block(x)
        return x


class Transformer_Decoder(nn.Module):
    """Decode fused temporal features into relative 2D displacements."""

    def __init__(
        self, input_dim, mlp_hid_dim, obs_len=16, num_head=1, drop_rate=0.0
    ):
        super(Transformer_Decoder, self).__init__()
        if input_dim % num_head != 0:
            raise ValueError("input_dim must be divisible by num_head")

        self.num_head = num_head
        self.q = nn.Linear(input_dim, input_dim, bias=False)
        self.kv = nn.Linear(input_dim, input_dim * 2, bias=False)
        self.attn_drop = nn.Dropout(p=drop_rate)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(p=drop_rate)
        self.mlp = Mlp(input_dim, mlp_hid_dim, 2)
        self.pos_emb = nn.Parameter(torch.zeros(1, obs_len, input_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        trunc_normal_(self.pos_emb, std=0.02)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, noise_input):
        if noise_input.size(1) != self.pos_emb.size(1):
            raise ValueError("input sequence length does not match configured obs_len")

        normalized = noise_input + self.pos_emb
        normalized = self.pos_drop(normalized)
        normalized = self.norm1(normalized)
        batch_size, sequence_length, feature_dim = normalized.shape

        q = self.q(normalized).reshape(
            batch_size,
            sequence_length,
            self.num_head,
            feature_dim // self.num_head,
        ).permute(0, 2, 1, 3)
        k_v = self.kv(normalized).reshape(
            batch_size,
            sequence_length,
            2,
            self.num_head,
            feature_dim // self.num_head,
        ).permute(2, 0, 3, 1, 4)
        k, v = k_v[0], k_v[1]

        attention = q @ k.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(
            batch_size, sequence_length, feature_dim
        )
        x = self.proj_drop(self.proj(x))
        x = self.norm2(x)
        return self.mlp(x)


class Trajectory_Generator(nn.Module):
    """Generate relative future positions from observed multi-agent tracks.

    The recovered architecture predicts the same number of steps as
    ``obs_len``. Agents are grouped by ``seq_start_end``. Within each group,
    pairwise relative positions are embedded and fused using learned sigmoid
    weights before stochastic decoding.
    """

    def __init__(
        self,
        obs_len,
        embedding_dim,
        encoder_input_dim,
        encoder_output_dim,
        encoder_mlp_dim,
        encoder_num_head,
        drop_rate,
        rel_traj_dim,
        noise_dim,
        merge_mlp_dim,
    ):
        super(Trajectory_Generator, self).__init__()
        if encoder_output_dim <= noise_dim:
            raise ValueError("noise_dim must be smaller than encoder_output_dim")

        self.obs_len = obs_len
        self.traj_embedding = Traj_embedding(embedding_dim)
        self.rel_embedding = Traj_embedding(rel_traj_dim)
        self.trans_encoder = Transformer_Encoder(
            encoder_input_dim,
            encoder_output_dim,
            encoder_mlp_dim,
            encoder_num_head,
            obs_len,
            drop_rate,
            num_block=2,
        )
        self.merge_mlp = Mlp(
            encoder_output_dim + rel_traj_dim,
            merge_mlp_dim,
            encoder_output_dim - noise_dim,
            drop_rate=drop_rate,
        )
        self.noise_dim = noise_dim
        self.trans_decoder = Transformer_Decoder(
            encoder_output_dim, merge_mlp_dim, obs_len
        )
        self.social_mlp = nn.Linear(
            (encoder_output_dim - noise_dim) * obs_len, 1
        )
        self.sigmoid = nn.Sigmoid()

        # Retained for checkpoint/state-dict compatibility with the historical
        # alternative convolutional fusion experiment.
        self.merge_conv = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)

    def add_noise(self, input_tensor, noise=None):
        """Append Gaussian or caller-supplied noise on the input device."""
        if self.noise_dim == 0:
            if noise is not None:
                raise ValueError("noise was provided but noise_dim is zero")
            return input_tensor

        expected_shape = (
            input_tensor.size(0),
            input_tensor.size(1),
            self.noise_dim,
        )
        if noise is None:
            noise = torch.randn(
                expected_shape,
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        elif tuple(noise.shape) != expected_shape:
            raise ValueError(
                "noise must have shape {}; got {}".format(
                    expected_shape, tuple(noise.shape)
                )
            )
        else:
            noise = noise.to(device=input_tensor.device, dtype=input_tensor.dtype)

        return torch.cat([input_tensor, noise], dim=-1)

    def _validate_inputs(self, obs_traj, obs_traj_rel, seq_start_end):
        if obs_traj.shape != obs_traj_rel.shape:
            raise ValueError("obs_traj and obs_traj_rel must have identical shapes")
        if obs_traj.dim() != 3 or obs_traj.size(-1) != 2:
            raise ValueError("trajectory input must have shape [time, agents, 2]")
        if obs_traj.size(0) != self.obs_len:
            raise ValueError("trajectory length does not match configured obs_len")
        if seq_start_end.dim() != 2 or seq_start_end.size(1) != 2:
            raise ValueError("seq_start_end must have shape [groups, 2]")
        if seq_start_end.size(0) == 0:
            raise ValueError("seq_start_end must contain at least one group")

        expected_start = 0
        for start_tensor, end_tensor in seq_start_end:
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            if start != expected_start or end <= start:
                raise ValueError("seq_start_end must be contiguous and non-empty")
            expected_start = end
        if expected_start != obs_traj.size(1):
            raise ValueError("seq_start_end must cover every agent exactly once")

    def _group_scales(self, obs_traj_by_agent, seq_start_end):
        """Reproduce the historical per-group x-range normalization.

        The original bytecode uses the x-coordinate span for both coordinates.
        That behavior is retained for compatibility and should be evaluated as
        part of the normalization ablation rather than silently changed.
        """
        scales = []
        for start_tensor, end_tensor in seq_start_end:
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            agent_count = end - start
            group = obs_traj_by_agent[start:end]
            max_xy = group.max(dim=0)[0].max(dim=0)[0]
            min_xy = group.min(dim=0)[0].min(dim=0)[0]
            scale = (max_xy - min_xy)[0] + 0.001
            scales.append(scale.repeat(self.obs_len, agent_count, 2))
        return torch.cat(scales, dim=1)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, noise=None):
        self._validate_inputs(obs_traj, obs_traj_rel, seq_start_end)
        obs_traj_by_agent = obs_traj.transpose(0, 1)
        group_scale = self._group_scales(obs_traj_by_agent, seq_start_end)

        trajectory_embedding = self.traj_embedding(obs_traj_rel / group_scale)
        trajectory_embedding = trajectory_embedding.transpose(0, 1)
        encoder_output = self.trans_encoder(trajectory_embedding)

        fused_groups = []
        for start_tensor, end_tensor in seq_start_end:
            start = int(start_tensor.item())
            end = int(end_tensor.item())
            agent_count = end - start
            group_embedding = encoder_output[start:end]
            group_obs_traj = obs_traj_by_agent[start:end]

            repeated_embedding = group_embedding.repeat(agent_count, 1, 1)
            repeated_positions_1 = group_obs_traj.repeat(agent_count, 1, 1)
            repeated_positions_2 = group_obs_traj.repeat(1, agent_count, 1)
            repeated_positions_2 = repeated_positions_2.view(
                agent_count * agent_count, self.obs_len, 2
            )

            relative_positions = repeated_positions_1 - repeated_positions_2
            relative_embedding = self.rel_embedding(relative_positions)
            pair_features = torch.cat(
                (repeated_embedding, relative_embedding), dim=2
            )
            pair_features = self.merge_mlp(pair_features)
            pair_features = pair_features.view(
                agent_count, agent_count, self.obs_len, -1
            )

            social_features = []
            for target_pair_features in pair_features:
                flattened = torch.flatten(target_pair_features, 1, 2)
                weights = self.sigmoid(self.social_mlp(flattened))
                weighted = weights.unsqueeze(1) * target_pair_features
                social_features.append(torch.sum(weighted, dim=0))

            fused_group = torch.cat(social_features, dim=0).view(
                agent_count, self.obs_len, -1
            )
            fused_groups.append(fused_group)

        fused = torch.cat(fused_groups, dim=0)
        decoder_input = self.add_noise(fused, noise=noise)
        pred_traj_rel = self.trans_decoder(decoder_input).transpose(0, 1)
        return pred_traj_rel * group_scale


class Classfier(nn.Module):
    """Historical discriminator head (name retained for compatibility)."""

    def __init__(self, input_dim, hid_dim, output_dim):
        super(Classfier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.act1 = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(hid_dim)
        self.fc2 = nn.Linear(hid_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.norm1(self.fc1(x)))
        return self.sigmoid(self.fc2(x))


class Trajectory_Discriminator(nn.Module):
    """Classify complete real or generated trajectories per agent."""

    def __init__(
        self,
        obs_len,
        embedding_dim,
        encoder_input_dim,
        encoder_output_dim,
        mlp_hid_dim,
        num_head,
        drop_rate,
    ):
        super(Trajectory_Discriminator, self).__init__()
        self.traj_embedding = Traj_embedding(embedding_dim)
        self.trans_encoder = Transformer_Encoder(
            encoder_input_dim,
            encoder_output_dim,
            mlp_hid_dim,
            num_head,
            obs_len,
            drop_rate,
        )
        self.real_classfier = Classfier(encoder_output_dim, 8, 1)

    def forward(self, pre_traj, seq_start_end):
        # seq_start_end is retained in the public API, although the historical
        # discriminator scores each agent independently.
        del seq_start_end
        embedded = self.traj_embedding(pre_traj).transpose(0, 1)
        encoded = self.trans_encoder(embedded).mean(dim=1)
        return self.real_classfier(encoded)


# Correctly spelled alias for new code; the historical misspelling remains the
# registered class name and therefore preserves source/checkpoint compatibility.
Classifier = Classfier

