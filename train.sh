#!/usr/bin/env bash

# QM9
# CUDA_VISIBLE_DEVICES=1 python train_mle.py --batch_size 256 \
#     --data_name qm9 \
#     --data_file qm9_relgcn_kekulized_ggnp.npz \
#     --epochs 10 \
#     --device cpu \
#     --num_node_masks 9 \
#     --num_channel_masks 9 \
#     --num_node_coupling 18 \
#     --num_channel_coupling 14 \
#     --additive_transformations True \
#     --apply_batch_norm True \
#     --node_mask_size 15 \
#     --mode train \
#     --num_gen 100 \
#     --show_loss_step 5 \
#     --gen_path ./results/qm9_mols \
#     --model_save_dir ./saved_qm9_models \
#     --img_dir=./results/qm9_img

# ZINC250k
CUDA_VISIBLE_DEVICES=1 python train_mle.py --batch_size 256 \
    --data_name zinc250k \
    --data_file zinc250k_relgcn_kekulized_ggnp.npz \
    --num_atoms 38 \
    --epochs 10 \
    --device cpu \
    --num_node_masks 19 \
    --num_channel_masks 19 \
    --num_node_coupling 18 \
    --num_channel_coupling 14 \
    --additive_transformations True \
    --apply_batch_norm True \
    --node_mask_size 15 \
    --mode train \
    --num_gen 100 \
    --show_loss_step 5 \
    --gen_path ./results/zinc_mols \
    --model_save_dir ./saved_zinc_models \
    --img_dir=./results/zinc_img

