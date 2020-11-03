#!/usr/bin/env bash


# # # SN gan
# export CUDA_VISIBLE_DEVICES=0
# /home/jathu/anaconda3/envs/sngan2/bin/python3 train.py \
# -gen_bs 128 \
# -dis_bs 64 \
# --dataset cifar10 \
# --img_size 32 \
# --max_iter 100000 \
# --model sngan_cifar10 \
# --latent_dim 128 \
# --gf_dim 256 \
# --df_dim 128 \
# --g_spectral_norm False \
# --d_spectral_norm True \
# --g_lr 0.0002 \
# --d_lr 0.0002 \
# --beta1 0.0 \
# --beta2 0.9 \
# --init_type xavier_uniform \
# --n_critic 5 \
# --val_freq 10 \
# --exp_name sngan








# # Resnet gan

# export CUDA_VISIBLE_DEVICES=0
# /home/jathu/anaconda3/envs/sngan2/bin/python3 train.py \
# -gen_bs 128 \
# -dis_bs 64 \
# --dataset cifar10 \
# --img_size 32 \
# --max_iter 100000 \
# --model sngan_cifar10 \
# --latent_dim 128 \
# --gf_dim 256 \
# --df_dim 128 \
# --g_spectral_norm False \
# --d_spectral_norm False \
# --g_lr 0.0002 \
# --d_lr 0.0002 \
# --beta1 0.0 \
# --beta2 0.9 \
# --init_type xavier_uniform \
# --n_critic 5 \
# --val_freq 10 \
# --exp_name resnetgan




# # # # wgan

export CUDA_VISIBLE_DEVICES=0
/home/jathu/anaconda3/envs/sngan2/bin/python3 train.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--img_size 32 \
--max_iter 100000 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm False \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 10 \
--exp_name wgan