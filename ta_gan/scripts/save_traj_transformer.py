import torch
from sgan.data.loader import data_loader
from sgan.models_transformer import Trajectory_Generator
from sgan.utils import get_dset_path, relative_to_abs
import argparse
import logging
import os
import sys
from sgan.utils import int_tuple, bool_flag
import matplotlib.pyplot as plt
import numpy
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='indoor_trajectory', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--obs_len', default=20, type=int)
parser.add_argument('--pred_len', default=20, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=1, type=int)
# parser.add_argument('--num_iterations', default=20000, type=int)
# parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
# parser.add_argument('--embedding_dim', default=16, type=int)
# parser.add_argument('--num_layers', default=1, type=int)
# parser.add_argument('--dropout', default=0, type=float)
# parser.add_argument('--batch_norm', default=0, type=bool_flag)
# parser.add_argument('--mlp_dim', default=64, type=int)

# Generator Options
# parser.add_argument('--encoder_h_dim_g', default=32, type=int)
# parser.add_argument('--decoder_h_dim_g', default=32, type=int)
# parser.add_argument('--noise_dim', default=(8,), type=int_tuple)
# parser.add_argument('--noise_type', default='gaussian')
# parser.add_argument('--noise_mix_type', default='global')
# parser.add_argument('--clipping_threshold_g', default=1.5, type=float)
# parser.add_argument('--g_learning_rate', default=1e-3, type=float)
# parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
# parser.add_argument('--pooling_type', default='pool_net')
# parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)

# Pool Net Option
# parser.add_argument('--bottleneck_dim', default=32, type=int)

# Social Pooling Options
# parser.add_argument('--neighborhood_size', default=2.0, type=float)
# parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
# parser.add_argument('--d_type', default='local', type=str)
# parser.add_argument('--encoder_h_dim_d', default=64, type=int)
# parser.add_argument('--d_learning_rate', default=1e-3, type=float)
# parser.add_argument('--d_steps', default=2, type=int)
# parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
# parser.add_argument('--l2_loss_weight', default=1, type=float)
# parser.add_argument('--best_k', default=10, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
# parser.add_argument('--print_every', default=50, type=int)
# parser.add_argument('--checkpoint_every', default=10, type=int)
# parser.add_argument('--checkpoint_name', default='gan_test')
# parser.add_argument('--checkpoint_start_from', default=None)
# parser.add_argument('--restore_from_checkpoint', default=0, type=int)
# parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def main(args):
    model = Trajectory_Generator(obs_len=args.obs_len,
                                 embedding_dim=16,
                                 encoder_input_dim=16,
                                 encoder_output_dim=16,
                                 encoder_mlp_dim=16,#64
                                 encoder_num_head=2,
                                 drop_rate=0,
                                 rel_traj_dim=16,
                                 noise_dim=4,
                                 merge_mlp_dim=16)#64
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load("best_model.pt"))
    val_path = get_dset_path(args.dataset_name, 'test')
    _, val_loader = data_loader(args, val_path)
    for batch in val_loader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch
        output = model(obs_traj, obs_traj_rel, seq_start_end)
        output = relative_to_abs(output, obs_traj[-1])
        output = torch.transpose(output, 0, 1).cpu().detach().numpy()
        obs_traj = torch.transpose(obs_traj, 0, 1).cpu().numpy()
        pred_traj_gt = torch.transpose(pred_traj_gt, 0, 1).cpu().numpy()
        output_x = output[:, :, 0]
        output_y = output[:, :, 1]
        obs_traj_x = obs_traj[:, :, 0]
        obs_traj_y = obs_traj[:, :, 1]
        pred_traj_gt_x = pred_traj_gt[:, :, 0]
        pred_traj_gt_y = pred_traj_gt[:, :, 1]
        for i in range(output.shape[0]):
            with open(f'./car_{i+1:d}.txt', 'a') as f:
                for j in range(args.obs_len):
                    f.write(str(output_x[i][j]))
                    f.write(' ')
                    f.write(str(output_y[i][j]))
                    f.write('\n')




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)