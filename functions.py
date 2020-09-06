# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import comet_ml
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import torch.nn.functional as F
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths

class Log_loss(torch.nn.Module):
    def __init__(self):
        # negation is true when you minimize -log(val)
        super(Log_loss, self).__init__()
       
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        log_val = torch.log(x)
        loss = torch.sum(log_val)
        if negation:
            loss = torch.neg(loss)
        return loss
    
class Itself_loss(torch.nn.Module):
    def __init__(self):
        super(Itself_loss, self).__init__()
        
    def forward(self, x, negation=True):
        # shape of x will be [batch size]
        loss = torch.sum(x)
        if negation:
            loss = torch.neg(loss)
        return loss


logger = logging.getLogger(__name__)

def train_d2(args, gen_net: nn.Module, dis_net1: nn.Module, dis_net2: nn.Module, gen_optimizer, dis_optimizer1, dis_optimizer2, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None, experiment=None):
    writer = writer_dict['writer']
    gen_step = 0

    criterion_log = Log_loss()
    criterion_itself = Itself_loss()

    # train mode
    gen_net = gen_net.train()
    dis_net1 = dis_net1.train()
    dis_net2 = dis_net2.train()

    d_loss1 = 0.0
    d_loss2 = 0.0
    g_loss = 0.0

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer1.zero_grad()
        dis_optimizer2.zero_grad()

        real_validity1 = dis_net1(real_imgs)
        real_validity2 = dis_net2(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity1 = dis_net1(fake_imgs.detach())
        fake_validity2 = dis_net2(fake_imgs.detach())

        d_loss1 = 0.2 * criterion_log(real_validity1) + criterion_itself(fake_validity1, False)
        d_loss1.backward()

        d_loss2 = criterion_itself(real_validity2, False) + 0.1*criterion_log(fake_validity2, False)
        d_loss2.backward()

        dis_optimizer1.step()
        dis_optimizer2.step()

        writer.add_scalar('d_loss1', d_loss1.item(), global_steps)
        writer.add_scalar('d_loss2', d_loss2.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity1 = dis_net1(gen_imgs)
            fake_validity2 = dis_net2(gen_imgs)

            # cal loss
            g_loss = criterion_itself(fake_validity1) + 0.1*criterion_log(fake_validity2)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            # if schedulers:
            #     gen_scheduler, dis_scheduler = schedulers
            #     g_lr = gen_scheduler.step(global_steps)
            #     d_lr = dis_scheduler.step(global_steps)
            #     writer.add_scalar('LR/g_lr', g_lr, global_steps)
            #     writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss1.item(), g_loss.item()))
            if experiment != None:
                experiment.log_metric("gen_loss", g_loss.item())
                experiment.log_metric("dis_loss1", d_loss1.item())
                experiment.log_metric("dis_loss2", d_loss2.item())

        writer_dict['train_global_steps'] = global_steps + 1


def train_chainer(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None, experiment=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    d_loss = 0.0
    g_loss = 0.0

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(F.softplus(-real_validity)) + \
                 torch.mean(F.softplus(fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = torch.mean(F.softplus(-fake_validity))
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                print("schedulars?")
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            # for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            #     avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))
            if experiment != None:
                experiment.log_metric("gen_loss", g_loss.item())
                experiment.log_metric("dis_loss", d_loss.item())

        writer_dict['train_global_steps'] = global_steps + 1

def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None, experiment=None):
    writer = writer_dict['writer']
    gen_step = 0

    criterion = nn.BCELoss() 

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    d_loss = 0.0
    g_loss = 0.0

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        y_real = torch.cuda.FloatTensor(torch.ones(imgs.shape[0], 1))

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
        y_fake = torch.cuda.FloatTensor(torch.zeros(real_imgs.size()[0], 1))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        # d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
        #          torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss = criterion(real_validity, y_real) + criterion(fake_validity, y_fake)
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)
            y_fake = torch.cuda.FloatTensor(torch.zeros(args.gen_batch_size, 1))

            # cal loss
            g_loss = criterion(fake_validity, y_fake)
            g_loss.backward()
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))
            if experiment != None:
                experiment.log_metric("gen_loss", g_loss.item())
                experiment.log_metric("dis_loss", d_loss.item())

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=8, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

    os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score, sample_imgs


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
