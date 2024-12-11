echo "1/6 Extracting vectors for santos"

python extractVectors.py --benchmark santos \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/santos_contrast_finetuned/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark santos \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/santos_contrast_lora/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark santos \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/santos_contrast_scratch/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark santos \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

########################################################

echo "2/6 Extracting vectors for TUS"

python extractVectors.py --benchmark tus \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tus_contrast_finetuned/checkpoints/epoch=08-validation_loss=0.1016.ckpt/checkpoint/mp_rank_00_model_states.pt
    
# python extractVectors.py --benchmark tus \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tus_contrast_lora/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark tus \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tus_contrast_scratch/checkpoints/epoch=09-validation_loss=1.0547.ckpt/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark tus \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

########################################################

echo "3/6 Extracting vectors for TUS LARGE"

python extractVectors.py --benchmark tusLarge \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tusLarge_contrast_finetuned/checkpoints/epoch=07-validation_loss=0.0266.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark tusLarge \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tusLarge_contrast_lora/checkpoints/epoch=09-validation_loss=0.6562-v1.ckpt/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark tusLarge \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/tusLarge_contrast_scratch/checkpoints/epoch=09-validation_loss=0.0422.ckpt/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark tusLarge \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

########################################################

echo "4/6 Extracting vectors for Pylon"

python extractVectors.py --benchmark pylon \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/pylon_contrast_finetuned/checkpoints/epoch=08-validation_loss=0.0021.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark pylon \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/pylon_contrast_lora/checkpoints/epoch=05-validation_loss=0.0269.ckpt/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark pylon \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/pylon_contrast_scratch/checkpoints/epoch=09-validation_loss=0.5791.ckpt/checkpoint/mp_rank_00_model_states.pt

python extractVectors.py --benchmark pylon \
    --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt










########################################################

# echo "5/6 Extracting vectors for ugen_v1"

# python extractVectors.py --benchmark ugen_v1 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v1_contrast_finetuned/checkpoints/epoch=09-validation_loss=0.0051.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v1 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v1_contrast_lora/checkpoints/epoch=04-validation_loss=0.0256.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v1 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v1_contrast_scratch/checkpoints/epoch=00-validation_loss=2.0156.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v1 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

########################################################

# echo "6/6 Extracting vectors for ugen_v2"

# python extractVectors.py --benchmark ugen_v2 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v2_contrast_finetuned/checkpoints/epoch=07-validation_loss=0.0299.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v2 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v2_contrast_lora/checkpoints/epoch=08-validation_loss=0.0386.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v2 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/ugen_v2_contrast_scratch/checkpoints/epoch=07-validation_loss=2.5781.ckpt/checkpoint/mp_rank_00_model_states.pt

# python extractVectors.py --benchmark ugen_v2 \
#     --checkpoint_dir /home/boutalebm/hytrel_vs_starmie/hypergraph-tabular-lm/checkpoints/contrast_pretrained/epoch=4-step=32690.ckpt/checkpoint/mp_rank_00_model_states.pt

