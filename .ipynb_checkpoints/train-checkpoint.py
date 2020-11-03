# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import comet_ml
comet_ml.config.save(api_key="jKV3NN31MK8hQrlT8N4sffY9f")

import cfg
import models
import datasets
from functions import train_multi, validate, LinearLrDecay, load_params, copy_params, train, train_wgan
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from torchvision.utils import make_grid


import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

import torchvision.utils as vutils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.'+args.model+'.Generator')(args=args).cuda()
#     dis_net1 = eval('models.'+args.model+'.Discriminator')(args=args).cuda()
    # dis_net2 = eval('models.'+args.model+'.Discriminator')(args=args).cuda()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

#     gen_net.apply(weights_init)
#     dis_net1.apply(weights_init)
    # dis_net2.apply(weights_init)
    
    multiD = []
    multiD_opt = []
    n_dis = 1
    for i in range(n_dis):
        dis_net = eval('models.'+args.model+'.Discriminator')(args=args).cuda()
#         dis_net.apply(weights_init)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                args.d_lr, (args.beta1, args.beta2))
        multiD.append(dis_net)
        multiD_opt.append(opt)
        
        
    print("XXXXXXX")
    print('Total params: %.2fM' % (sum(p.numel() for p in multiD[0].parameters()) / 1000000.0))
    print('Total params: %.2fM' % (sum(p.numel() for p in gen_net.parameters()) / 1000000.0))


    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer1 = torch.optim.Adam(filter(lambda p: p.requires_grad, multiD[0].parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    # dis_optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net2.parameters()),
    #                                  args.d_lr, (args.beta1, args.beta2))                              
    # gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    # dis_scheduler1 = LinearLrDecay(dis_optimizer1, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)
    # dis_scheduler2 = LinearLrDecay(dis_optimizer2, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

#     # fid stat
    fid_stat = None
#     if args.dataset.lower() == 'cifar10':
#         fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
#     elif args.dataset.lower() == 'stl10':
#         fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
#     else:
#         raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
#     assert os.path.exists(fid_stat)
#     

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (64, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_incept = 0
    
    alpha2 = 1.5
    t2 = 10
    
    print("XXXXXXXX", args.alpha)

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
#         dis_net1.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer1.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    experiment = comet_ml.Experiment(project_name="cub-sngan")
    exp_parameters = {
        "data": "cub_32x32",
        "model": "sngan-cifar10",
        "opt_gen": "Adam_lr_0.0002, (0.0,0.999)",
        "opt_dis": "Adam_lr_0.0002, (0.0,0.999)",
#         "alpha": "0.0,2",
        "freq": 20,
        # "gp lamba": 10,
        "rand_thresh": 0.7,
        "n_dis": n_dis,
        "z_dim": 128,
#         "n_critic": 7,
        "normalize": "mean,std 0.5",
        "dis_landscape": 0,
        "try": 0,
        "model_save": args.path_helper['log_path']
    }
    output = '.temp_cub.png'
    experiment.log_parameters(exp_parameters)

    # experiment = None

    # train loop
    lr_schedulers = None#(gen_scheduler, dis_scheduler1) if args.lr_decay else None
    print("args.lr_decay: ", args.lr_decay)
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        multiD, multiD_opt, t2, alpha2 = train_multi(args, gen_net, multiD, gen_optimizer, multiD_opt, gen_avg_param, train_loader, epoch, writer_dict, alpha_m=args.alpha, t=args.t, check_ep=t2, alpha=alpha2, schedulers=None, experiment=experiment)       # SN, Resnet  
#         train_wgan(args, gen_net, multiD, gen_optimizer, multiD_opt, gen_avg_param, train_loader, epoch, writer_dict,lr_schedulers, experiment)       # wgan

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
#             backup_param = copy_params(gen_net)
#             load_params(gen_net, gen_avg_param)
            fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (64, args.latent_dim)))
#             inception_score, fid_score, sample_imgs = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            inception_score, fid_score = 0,0 
    
            with torch.no_grad():
                sample_imgs = gen_net(fixed_z)
            img_grid = make_grid(sample_imgs, nrow=8, normalize=True, scale_each=True)

            logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
            
            vutils.save_image(sample_imgs, output ,normalize=True)
            experiment.log_image(output, name = "output_" + str(epoch))
            experiment.log_metric("IS", inception_score)
            experiment.log_metric("FID", fid_score)
            experiment.log_metric("n_dis", len(multiD))
            
            del(sample_imgs)
            del(fixed_z)
            
#             load_params(gen_net, backup_param)
            if inception_score > best_incept:
                best_incept = inception_score
                is_best = True
            else:
                is_best = False
                
                
                
            print(inception_score, fid_score)
        else:
            is_best = False

        if is_best:
            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'gen_state_dict': gen_net.state_dict(),
#                 'dis1_state_dict': dis_net1.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer1.state_dict(),
                'best_fid': best_fid,
                'best_incept': best_incept,
                'path_helper': args.path_helper
            }, is_best, args.path_helper['ckpt_path'])
            del avg_gen_net


if __name__ == '__main__':
    main()
