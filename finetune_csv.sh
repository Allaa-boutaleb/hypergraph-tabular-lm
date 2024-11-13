#!/bin/bash

# This script is for finetuning HyTrel on SANTOS data with contrastive learning
# Make sure to run prepare_santos_data.py first

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Adjust based on available GPUs

# Run finetuning with modified parameters for SANTOS
python -W ignore run_pretrain.py \
    --data_path './data/pretrain' \
    --contrast_bipartite_edge True \
    --gradient_clip_val 2.0 \
    --accelerator "gpu" \
    --devices 1 \
    --max_epochs 1 \
    --batch_size 256 \
    --base_learning_rate 1e-5 \
    --accumulate_grad_batches 4 \
    --replace_sampler_ddp False \
    --checkpoint_path "./checkpoints/contrast/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt" \
    --warmup_step_ratio 0.1


