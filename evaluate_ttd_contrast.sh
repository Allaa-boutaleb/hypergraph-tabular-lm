# This script is tested on a G5 AWS instance with A10 GPUs; 
# Please change it according to your own environment.
CUDA_VISIBLE_DEVICES=0 python -W ignore evaluate_ttd.py --data_path './data/ttd/' --checkpoint_path "./checkpoints/contrast/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt"  --max_epochs 10 --gradient_clip_val 2.0 --accelerator "gpu" --devices 1
        
