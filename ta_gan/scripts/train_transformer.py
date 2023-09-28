import argparse
import gc
import logging
import os
import sys
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses import speed_penalty_term
from sgan.losses import displacement_error, final_displacement_error

from sgan.models_transformer import Trajectory_Generator, Trajectory_Discriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path
import numpy as np
#torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=20000, type=int)
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=64, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)
parser.add_argument('--decoder_h_dim_g', default=32, type=int)
parser.add_argument('--noise_dim', default=(8,), type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')
parser.add_argument('--clipping_threshold_g', default=1.5, type=float)
parser.add_argument('--g_learning_rate', default=1e-3, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=1e-3, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)
parser.add_argument('--best_k', default=10, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=50, type=int)
parser.add_argument('--checkpoint_every', default=10, type=int)
parser.add_argument('--checkpoint_name', default='gan_test')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_path = get_dset_path(args.dataset_name, 'train')
    val_path = get_dset_path(args.dataset_name, 'test')
    print(train_path)
    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)
    generator = Trajectory_Generator(obs_len=args.obs_len,
                                     embedding_dim=16,
                                     encoder_input_dim=16,
                                     encoder_output_dim=16,
                                     encoder_mlp_dim=16,
                                     encoder_num_head=2,
                                     drop_rate=0,
                                     rel_traj_dim=16,
                                     noise_dim=4,
                                     merge_mlp_dim=16
        )


    generator.apply(init_weights)
    generator.type(float_dtype)
    logger.info('Here is the generator:')
    logger.info(generator)
    g_loss = torch.nn.MSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    min_ade = 1000
    min_fde = 1000
    for epoch in range(args.num_epochs):
        logger.info("epoch:{} starting-----------------------------".format(epoch))
        loss_total = 0
        count = 0
        generator.train()
        for batch in train_loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
            pred_traj = relative_to_abs(generator_out, obs_traj[-1])
            loss = g_loss(pred_traj, pred_traj_gt) #+ 0.5 * speed_penalty_term(generator_out, obs_traj)
            loss_total += loss
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            count += 1

        loss_mean = loss_total / count
        logger.info("train loss:{:.3f}".format(loss_mean))
        ade, fde = val(generator, val_loader, logger)
        if ade < min_ade:
            min_ade = ade
            min_fde = fde
            torch.save(generator.state_dict(), 'best_model.pt')
    logger.info("min ade:{:.3f}".format(min_ade))
    logger.info("min fde:{:.3f}".format(min_fde))

def val(generator, val_dataloader, logger):
    g_loss = torch.nn.MSELoss()
    generator.eval()
    loss_total = 0
    count = 0
    total_traj = 0
    disp_error = []
    f_disp_error = []
    for batch in val_dataloader:
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
         loss_mask, seq_start_end) = batch
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
        pred_traj = relative_to_abs(generator_out, obs_traj[-1])
        loss = g_loss(pred_traj, pred_traj_gt)
        ade = displacement_error(pred_traj, pred_traj_gt)
        fde = final_displacement_error(pred_traj[-1], pred_traj_gt[-1])
        disp_error.append(ade.item())
        f_disp_error.append(fde.item())
        loss_total += loss
        count += 1
        total_traj += pred_traj_gt.size(1)
    ade = sum(disp_error) / (total_traj * args.pred_len)#.....args.pred_len注意可能更改
    fde = sum(f_disp_error) / total_traj
    loss_mean = loss_total / count
    logger.info("val loss:{:.3f}".format(loss_mean))
    logger.info("val ade:{:.3f}".format(ade))
    logger.info("val fde:{:.3f}".format(fde))
    return ade, fde





if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
