"""Social-GAN baseline models used by the TA-GAN evaluation scripts.

Provenance
----------
This module restores the missing source corresponding to
``sgan/__pycache__/models.cpython-36.pyc``.  The retained bytecode was compared
with Social-GAN commit ``691231e1adb6a344c7bcea9ebf2534518b226ead``.  All
functions match that upstream implementation except that the retained
``Encoder.forward`` calls ``contiguous()`` before ``view()``; that local change
is preserved below.

The only intentional modernization is device-safe temporary tensor creation.
The historical code called ``.cuda()`` directly, which prevented CPU testing
and failed when inputs and the default CUDA device differed.  These changes do
not add parameters or alter state-dict names and shapes.
"""

import torch
import torch.nn as nn


def make_mlp(dim_list, activation="relu", batch_norm=True, dropout=0):
    """Build the feed-forward blocks shared by pooling and classification."""
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type, reference=None):
    """Sample noise on the same device and with the same dtype as reference."""
    kwargs = {}
    if reference is not None:
        kwargs = {"device": reference.device, "dtype": reference.dtype}
    if noise_type == "gaussian":
        return torch.randn(*shape, **kwargs)
    if noise_type == "uniform":
        return torch.rand(*shape, **kwargs).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """LSTM encoder shared by the trajectory generator and discriminator."""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        dropout=0.0,
    ):
        super(Encoder, self).__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch, reference=None):
        """Return zero LSTM state on the model/input device."""
        if reference is None:
            reference = self.spatial_embedding.weight
        state = reference.new_zeros(self.num_layers, batch, self.h_dim)
        return state, state.clone()

    def forward(self, obs_traj):
        """Encode relative coordinates shaped ``(obs_len, batch, 2)``."""
        batch = obs_traj.size(1)
        # ``contiguous`` is the sole known TA-GAN-local change from Social-GAN.
        obs_traj_embedding = self.spatial_embedding(
            obs_traj.contiguous().view(-1, 2)
        )
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch, obs_traj_embedding)
        _, state = self.encoder(obs_traj_embedding, state_tuple)
        return state[0]


class Decoder(nn.Module):
    """Autoregressive LSTM decoder used by :class:`TrajectoryGenerator`."""

    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        pooling_type="pool_net",
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        if pool_every_timestep:
            if pooling_type == "pool_net":
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            elif pooling_type == "spool":
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size,
                )
            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """Predict relative positions for ``self.seq_len`` future steps."""
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat(
                    [decoder_h.view(-1, self.h_dim), pool_h], dim=1
                )
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            decoder_input = self.spatial_embedding(rel_pos)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        return torch.stack(pred_traj_fake_rel, dim=0), state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pairwise max-pooling module proposed by Social-GAN."""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super(PoolHiddenNet, self).__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        mlp_pre_pool_dims = [embedding_dim + h_dim, 512, bottleneck_dim]
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    @staticmethod
    def repeat(tensor, num_reps):
        """Repeat rows as ``R1, R1, R2, R2``."""
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        return tensor.view(-1, col_len)

    def forward(self, h_states, seq_start_end, end_pos):
        """Pool interactions independently inside each scene partition."""
        pool_h = []
        for start, end in seq_start_end:
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_embedding = self.spatial_embedding(
                curr_end_pos_1 - curr_end_pos_2
            )
            mlp_h_input = torch.cat(
                [curr_rel_embedding, curr_hidden_1], dim=1
            )
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        return torch.cat(pool_h, dim=0)


class SocialPooling(nn.Module):
    """Grid pooling from Social-LSTM (CVPR 2016)."""

    def __init__(
        self,
        h_dim=64,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        neighborhood_size=2.0,
        grid_size=8,
        pool_dim=None,
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        output_dim = pool_dim if pool_dim else h_dim
        mlp_pool_dims = [grid_size * grid_size * h_dim, output_dim]
        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            (other_pos[:, 0] - top_left[:, 0])
            / self.neighborhood_size
            * self.grid_size
        )
        cell_y = torch.floor(
            (top_left[:, 1] - other_pos[:, 1])
            / self.neighborhood_size
            * self.grid_size
        )
        return cell_x + cell_y * self.grid_size

    @staticmethod
    def repeat(tensor, num_reps):
        """Repeat rows as ``R1, R1, R2, R2``."""
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        return tensor.view(-1, col_len)

    def forward(self, h_states, seq_start_end, end_pos):
        """Pool hidden states into spatial grids for every pedestrian."""
        pool_h = []
        for start, end in seq_start_end:
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = num_ped * grid_size + 1
            curr_pool_h = curr_hidden.new_zeros(
                (curr_pool_h_size, self.h_dim)
            )
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)
            grid_pos = self.get_grid_locations(
                top_left, curr_end_pos
            ).type_as(seq_start_end)

            x_bound = (curr_end_pos[:, 0] >= bottom_right[:, 0]) + (
                curr_end_pos[:, 0] <= top_left[:, 0]
            )
            y_bound = (curr_end_pos[:, 1] >= top_left[:, 1]) + (
                curr_end_pos[:, 1] <= bottom_right[:, 1]
            )
            within_bound = x_bound + y_bound
            within_bound[0 :: num_ped + 1] = 1
            within_bound = within_bound.view(-1)

            # Index zero is a sink for self-interactions and out-of-grid pairs.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0,
                total_grid_size * num_ped,
                total_grid_size,
                device=seq_start_end.device,
            ).type_as(seq_start_end)
            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)
            curr_pool_h = curr_pool_h.scatter_add(
                0, grid_pos, curr_hidden_repeat
            )
            pool_h.append(curr_pool_h[1:].view(num_ped, -1))

        return self.mlp_pool(torch.cat(pool_h, dim=0))


class TrajectoryGenerator(nn.Module):
    """Social-GAN trajectory generator retained as a TA-GAN baseline."""

    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        encoder_h_dim=64,
        decoder_h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        noise_dim=(0,),
        noise_type="gaussian",
        noise_mix_type="ped",
        pooling_type=None,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(TrajectoryGenerator, self).__init__()
        if pooling_type and pooling_type.lower() == "none":
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        # Retained for checkpoint/API compatibility with upstream Social-GAN.
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
        )

        if pooling_type == "pool_net":
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
            )
        elif pooling_type == "spool":
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size,
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        input_dim = (
            encoder_h_dim + bottleneck_dim
            if pooling_type
            else encoder_h_dim
        )
        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim,
                mlp_dim,
                decoder_h_dim - self.noise_first_dim,
            ]
            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

    def add_noise(self, model_input, seq_start_end, user_noise=None):
        """Append per-pedestrian or per-scene latent noise to decoder state."""
        if not self.noise_dim:
            return model_input

        if self.noise_mix_type == "global":
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (model_input.size(0),) + self.noise_dim

        if user_noise is None:
            z_decoder = get_noise(
                noise_shape, self.noise_type, reference=model_input
            )
        else:
            if tuple(user_noise.shape) != tuple(noise_shape):
                raise ValueError(
                    "user_noise must have shape %s, got %s"
                    % (tuple(noise_shape), tuple(user_noise.shape))
                )
            z_decoder = user_noise.to(
                device=model_input.device, dtype=model_input.dtype
            )

        if self.noise_mix_type == "global":
            scene_states = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                scene_noise = z_decoder[idx].view(1, -1).repeat(
                    end - start, 1
                )
                scene_states.append(
                    torch.cat(
                        [model_input[start:end], scene_noise], dim=1
                    )
                )
            return torch.cat(scene_states, dim=0)

        return torch.cat([model_input, z_decoder], dim=1)

    def mlp_decoder_needed(self):
        return bool(
            self.noise_dim
            or self.pooling_type
            or self.encoder_h_dim != self.decoder_h_dim
        )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """Generate relative future coordinates shaped ``(pred_len, N, 2)``."""
        batch = obs_traj_rel.size(1)
        final_encoder_h = self.encoder(obs_traj_rel)

        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h],
                dim=1,
            )
        else:
            context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(context_input)
        else:
            noise_input = context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise
        ).unsqueeze(0)
        decoder_c = decoder_h.new_zeros(
            self.num_layers, batch, self.decoder_h_dim
        )
        pred_traj_fake_rel, _ = self.decoder(
            obs_traj[-1],
            obs_traj_rel[-1],
            (decoder_h, decoder_c),
            seq_start_end,
        )
        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    """Classify complete trajectories using local or pooled scene context."""

    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        d_type="local",
    ):
        super(TrajectoryDiscriminator, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.real_classifier = make_mlp(
            [h_dim, mlp_dim, 1],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        if d_type == "global":
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=[h_dim + embedding_dim, mlp_dim, h_dim],
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """Return one unbounded real/fake score per trajectory."""
        final_h = self.encoder(traj_rel)
        if self.d_type == "local":
            classifier_input = final_h.squeeze()
        else:
            # Historical Social-GAN behavior pools against start positions.
            classifier_input = self.pool_net(
                final_h.squeeze(), seq_start_end, traj[0]
            )
        return self.real_classifier(classifier_input)
