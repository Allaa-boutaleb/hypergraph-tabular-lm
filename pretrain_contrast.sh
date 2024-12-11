#!/bin/bash

# Base parameters for all runs
BASE_PARAMS="--contrast_bipartite_edge True \
    --accelerator gpu \
    --devices 1 \
    --replace_sampler_ddp False \
    --accumulate_grad_batches 4 \
    --batch_size 16 \
    --max_epoch 10 \
    --save_every_n_epochs 1 \
    --save_top_k 1"

# Checkpoint path for finetuning
BASE_CHECKPOINT="./checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt"

# echo "Starting training sequence..."
# echo "=========================================="

# # Santos Dataset
# echo "Processing Santos Dataset..."

echo "1/3: Training Santos from scratch..."
CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
    --data_path './data/santos/' \
    --gradient_clip_val 2.0 \
    --base_learning_rate 5e-5 \
    --checkpoint_dir './checkpoints/santos_contrast_scratch' \
    $BASE_PARAMS

# echo "2/3: Finetuning Santos (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/santos/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/santos_contrast_finetuned' \
#     $BASE_PARAMS

# echo "3/3: Finetuning Santos with LoRA..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/santos/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/santos_contrast_lora' \
#     --use_lora True \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     $BASE_PARAMS

# echo "Santos Dataset Complete"
# echo "=========================================="

# TUS Dataset
# echo "Processing TUS Dataset..."

# echo "1/3: Training TUS from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tus/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --checkpoint_dir './checkpoints/tus_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/3: Finetuning TUS (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tus/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/tus_contrast_finetuned' \
#     $BASE_PARAMS

# echo "3/3: Finetuning TUS with LoRA..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tus/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/tus_contrast_lora' \
#     --use_lora True \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     $BASE_PARAMS

# echo "TUS Dataset Complete"
# echo "=========================================="


# # TUS LARGE Dataset
# echo "Processing TUS LARGE Dataset..."

# echo "1/3: Training TUS LARGE from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tusLarge/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --checkpoint_dir './checkpoints/tusLarge_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/3: Finetuning TUS LARGE (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tusLarge/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/tusLarge_contrast_finetuned' \
#     $BASE_PARAMS

# echo "3/3: Finetuning TUS LARGE with LoRA..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/tusLarge/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/tusLarge_contrast_lora' \
#     --use_lora True \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     $BASE_PARAMS

# echo "TUS LARGE Dataset Complete"
# echo "=========================================="

# # Pylon Dataset
# echo "Processing Pylon Dataset..."

# echo "1/3: Training Pylon from scratch..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/pylon/' \
#     --gradient_clip_val 2.0 \
#     --base_learning_rate 5e-5 \
#     --checkpoint_dir './checkpoints/pylon_contrast_scratch' \
#     $BASE_PARAMS

# echo "2/3: Finetuning Pylon (classic)..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/pylon/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/pylon_contrast_finetuned' \
#     $BASE_PARAMS

# echo "3/3: Finetuning Pylon with LoRA..."
# CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
#     --data_path './data/pylon/' \
#     --gradient_clip_val 1.0 \
#     --base_learning_rate 1e-5 \
#     --checkpoint_path "$BASE_CHECKPOINT" \
#     --checkpoint_dir './checkpoints/pylon_contrast_lora' \
#     --use_lora True \
#     --lora_r 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0.1 \
#     $BASE_PARAMS

# echo "Pylon Dataset Complete"
# echo "=========================================="

# # # UGEN_V1 Dataset
# # echo "Processing UGEN_V1 Dataset..."

# # echo "1/3: Training UGEN_V1 from scratch..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v1/' \
# #     --gradient_clip_val 2.0 \
# #     --base_learning_rate 5e-5 \
# #     --checkpoint_dir './checkpoints/ugen_v1_contrast_scratch' \
# #     $BASE_PARAMS

# # echo "2/3: Finetuning UGEN_V1 (classic)..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v1/' \
# #     --gradient_clip_val 1.0 \
# #     --base_learning_rate 1e-5 \
# #     --checkpoint_path "$BASE_CHECKPOINT" \
# #     --checkpoint_dir './checkpoints/ugen_v1_contrast_finetuned' \
# #     $BASE_PARAMS

# # echo "3/3: Finetuning UGEN_V1 with LoRA..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v1/' \
# #     --gradient_clip_val 1.0 \
# #     --base_learning_rate 1e-5 \
# #     --checkpoint_path "$BASE_CHECKPOINT" \
# #     --checkpoint_dir './checkpoints/ugen_v1_contrast_lora' \
# #     --use_lora True \
# #     --lora_r 8 \
# #     --lora_alpha 16 \
# #     --lora_dropout 0.1 \
# #     $BASE_PARAMS

# # echo "UGEN_V1 Dataset Complete"
# echo "=========================================="

# # # UGEN_V2 Dataset
# # echo "Processing UGEN_V2 Dataset..."

# # echo "1/3: Training UGEN_V2 from scratch..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v2/' \
# #     --gradient_clip_val 2.0 \
# #     --base_learning_rate 5e-5 \
# #     --checkpoint_dir './checkpoints/ugen_v2_contrast_scratch' \
# #     $BASE_PARAMS

# # echo "2/3: Finetuning UGEN_V2 (classic)..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v2/' \
# #     --gradient_clip_val 1.0 \
# #     --base_learning_rate 1e-5 \
# #     --checkpoint_path "$BASE_CHECKPOINT" \
# #     --checkpoint_dir './checkpoints/ugen_v2_contrast_finetuned' \
# #     $BASE_PARAMS

# # echo "3/3: Finetuning UGEN_V2 with LoRA..."
# # CUDA_VISIBLE_DEVICES=0 python -W ignore run_pretrain.py \
# #     --data_path './data/ugen_v2/' \
# #     --gradient_clip_val 1.0 \
# #     --base_learning_rate 1e-5 \
# #     --checkpoint_path "$BASE_CHECKPOINT" \
# #     --checkpoint_dir './checkpoints/ugen_v2_contrast_lora' \
# #     --use_lora True \
# #     --lora_r 8 \
# #     --lora_alpha 16 \
# #     --lora_dropout 0.1 \
# #     $BASE_PARAMS

# # echo "UGEN_V2 Dataset Complete"
# echo "=========================================="

# echo "All training completed!"
# echo "For each dataset, you now have:"
# echo "- *_scratch: Trained from scratch"
# echo "- *_finetuned: Classic finetuning"
# echo "- *_lora: LoRA finetuning"