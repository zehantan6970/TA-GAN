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
from torch.autograd import Variable
from sgan.data.loader import data_loader
#from sgan.losses import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models_transformer import Trajectory_Generator, Trajectory_Discriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)#indoor_trajectory
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=8, type=int)
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
parser.add_argument('--d_learning_rate', default=1e-4, type=float)
parser.add_argument('--d_steps', default=1, type=int)
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
    val_path = get_dset_path(args.dataset_name, 'val')
    test_path = get_dset_path(args.dataset_name, 'test')

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    iterations = len(train_dset)
    #print(iterations)
    # logger.info("Initializing val dataset")
    # _, val_loader = data_loader(args, val_path)
    logger.info("Initializing test dataset")
    _, test_loader = data_loader(args, test_path)

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
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = Trajectory_Discriminator(
        obs_len=args.obs_len * 2,
        embedding_dim=8,
        encoder_input_dim=8,
        encoder_output_dim=8,
        mlp_hid_dim=8,
        num_head=2,
        drop_rate=0
        )
    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = nn.BCELoss()
    d_loss_fn = nn.BCELoss()
    g_l2_loss = nn.MSELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )
    min_ade = 1000
    min_fde = 1000
    for epoch in range(args.num_epochs):
        logger.info("epoch:{} starting-----------------------------".format(epoch))
        for d_step in range(args.d_steps):
            discriminator.train()
            generator.eval()
            D_loss = 0
            D_count = 0
            D_real_loss = 0
            D_fake_loss = 0
            for batch in train_loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
                 loss_mask, seq_start_end) = batch
                generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
                generator_out = relative_to_abs(generator_out, obs_traj[-1])
                fake_traj = torch.cat([obs_traj, generator_out], dim=0)
                real_traj = torch.cat([obs_traj, pred_traj_gt], dim=0)
                score_fake = discriminator(fake_traj, seq_start_end)
                score_real = discriminator(real_traj, seq_start_end)
                score_fake_label = Variable(torch.zeros(obs_traj.size(1), 1)).cuda()
                score_real_label = Variable(torch.ones(obs_traj.size(1), 1)).cuda()
                d_loss_fake = d_loss_fn(score_fake, score_fake_label)
                d_loss_real = d_loss_fn(score_real, score_real_label)
                d_loss = d_loss_real + d_loss_fake
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()
                D_loss += d_loss.item()
                D_real_loss += d_loss_real.item()
                D_fake_loss += d_loss_fake.item()
                D_count += 1
            D_loss_mean = D_loss / D_count
            D_real_loss_mean = D_real_loss / D_count
            D_fake_loss_mean = D_fake_loss / D_count
            logger.info("[D] loss:{:.3f}".format(D_loss_mean))
            logger.info("[D real] loss:{:.3f}".format(D_real_loss_mean))
            logger.info("[D fake] loss:{:.3f}".format(D_fake_loss_mean))

        for g_step in range(args.g_steps):
            generator.train()
            #discriminator.eval()
            G_bce_loss = 0
            G_l2_loss = 0
            G_count = 0
            for batch in train_loader:
                batch = [tensor.cuda() for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
                 loss_mask, seq_start_end) = batch
                generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
                generator_out = relative_to_abs(generator_out, obs_traj[-1])
                g_real_traj = torch.cat([obs_traj, generator_out], dim=0)
                score = discriminator(g_real_traj, seq_start_end)
                g_real_label = Variable(torch.ones(obs_traj.size(1), 1).cuda())
                g_bce_loss = g_loss_fn(score, g_real_label) #+ args.l2_loss_weight * g_l2_loss(generator_out, pred_traj_gt)
                l2_loss = g_l2_loss(generator_out, pred_traj_gt)
                g_loss = g_bce_loss + args.l2_loss_weight * l2_loss
                optimizer_g.zero_grad()
                g_loss.backward()
                optimizer_g.step()
                G_bce_loss += g_bce_loss.item()
                G_l2_loss += l2_loss.item()
                G_count += 1
            G_bce_loss_mean = G_bce_loss / G_count
            G_l2_loss_mean = G_l2_loss / G_count
            logger.info("[G bce] loss:{:.3f}".format(G_bce_loss_mean))
            logger.info("[G l2] loss:{:.3f}".format(G_l2_loss_mean))

        #_, _ = val(generator, val_loader, logger)
        ade, fde = test(generator, test_loader, logger)
        if ade < min_ade:
            min_ade = ade
            min_fde = fde
            torch.save(generator.state_dict(), 'best_gan_model.pt')
    logger.info("min ade:{:.3f}".format(min_ade))
    logger.info("min fde:{:.3f}".format(min_fde))
    torch.save(generator.state_dict(), 'last_gan_model.pt')

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
        generator_out = relative_to_abs(generator_out, obs_traj[-1])
        loss = g_loss(generator_out, pred_traj_gt)
        ade = displacement_error(generator_out, pred_traj_gt)
        fde = final_displacement_error(generator_out[-1], pred_traj_gt[-1])
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

def test(generator, test_dataloader, logger):

    generator.eval()
    count = 0
    total_traj = 0
    disp_error = []
    f_disp_error = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)
            generator_out = relative_to_abs(generator_out, obs_traj[-1])

            ade = displacement_error(generator_out, pred_traj_gt)
            fde = final_displacement_error(generator_out[-1], pred_traj_gt[-1])
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())
            count += 1
            total_traj += pred_traj_gt.size(1)
    ade = sum(disp_error) / (total_traj * args.pred_len)  # .....args.pred_len注意可能更改
    fde = sum(f_disp_error) / total_traj

    logger.info("test ade:{:.3f}".format(ade))
    logger.info("test fde:{:.3f}".format(fde))
    return ade, fde

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
