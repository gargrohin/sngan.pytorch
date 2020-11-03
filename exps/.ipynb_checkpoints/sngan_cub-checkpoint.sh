#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
/home/jathu/anaconda3/envs/sngan2/bin/python3 train.py \
-gen_bs 16 \
-dis_bs 16 \
--dataset cub \
--img_size 128 \
--max_iter 100000 \
--model sngan_cifar10_2 \
--latent_dim 128 \
--gf_dim 64 \
--df_dim 64 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 1 \
--exp_name sngan-cub \
--alpha 1.7 \
--t 2.5
