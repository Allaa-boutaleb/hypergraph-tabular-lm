# #!/bin/bash

python evaluate_benchmark.py santos
python evaluate_benchmark.py santos --distances_only

python evaluate_benchmark.py pylon
python evaluate_benchmark.py pylon --distances_only

python evaluate_benchmark.py tus
python evaluate_benchmark.py tus --distances_only

python evaluate_benchmark.py tusLarge
python evaluate_benchmark.py tusLarge --distances_only

# echo "Starting Santos benchmark evaluation..."
# python test_naive_search.py \
#     --encoder cl \
#     --benchmark santos \
#     --augment_op drop_col \
#     --sample_meth tfidf_entity \
#     --table_order column \
#     --run_id 0 \
#     --K 10 \
#     --variant original \
#     --compare_original

# echo "Starting TUS benchmark evaluation..."
# python test_naive_search.py \
#     --encoder cl \
#     --benchmark tus \
#     --augment_op drop_cell \
#     --sample_meth alphaHead \
#     --table_order column \
#     --run_id 0 \
#     --K 60 \
#     --variant original \
#     --compare_original

# echo "Starting TUSLarge benchmark evaluation..."
# python test_naive_search.py \
#     --encoder cl \
#     --benchmark tusLarge \
#     --augment_op drop_cell \
#     --sample_meth alphaHead \
#     --table_order column \
#     --run_id 0 \
#     --K 60 \
#     --variant original \
#     --compare_original

# echo "Starting Pylon benchmark evaluation..."
# python test_naive_search.py \
#     --encoder cl \
#     --benchmark pylon \
#     --augment_op drop_col \
#     --sample_meth tfidf_entity \
#     --table_order column \
#     --run_id 0 \
#     --K 10 \
#     --variant original \
#     --compare_original

# echo "All benchmark evaluations completed!"