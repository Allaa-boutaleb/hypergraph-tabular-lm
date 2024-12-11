import os
import shutil
import logging
import pickle
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_file_integrity(src_path, dst_path):
    """Verify files have same size and can be loaded"""
    if os.path.getsize(src_path) != os.path.getsize(dst_path):
        return False
    try:
        with open(src_path, 'rb') as f:
            pickle.load(f)
        with open(dst_path, 'rb') as f:
            pickle.load(f)
        return True
    except:
        return False

def reorganize_embeddings(benchmark):
    """Reorganize embeddings for a given benchmark"""
    logging.info(f"Processing benchmark: {benchmark}")
    
    # Setup paths
    vectors_dir = Path("vectors")
    benchmark_dir = vectors_dir / benchmark
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model types
    model_types = ['finetuned', 'lora', 'pretrained', 'scratch']
    
    # Process original benchmark embeddings
    src_base = Path(f"data/{benchmark}/vectors")
    if src_base.exists():
        # Process both query and datalake embeddings for each model type
        for model_type in model_types:
            # Handle query embeddings (only for original variant)
            query_src = src_base / f"query_vectors_{model_type}.pkl"
            if query_src.exists():
                query_dst = benchmark_dir / f"hytrel_query_embeddings_{model_type}.pkl"
                if not query_dst.exists():
                    logging.info(f"Copying {query_src} to {query_dst}")
                    shutil.copy2(query_src, query_dst)
                    
                    if verify_file_integrity(query_src, query_dst):
                        logging.info(f"Successfully verified: {query_dst}")
                    else:
                        logging.error(f"File verification failed: {query_dst}")
                        os.remove(query_dst)
            
            # Handle datalake embeddings
            datalake_src = src_base / f"datalake_vectors_{model_type}.pkl"
            if datalake_src.exists():
                datalake_dst = benchmark_dir / f"hytrel_datalake_embeddings_{model_type}.pkl"
                if not datalake_dst.exists():
                    logging.info(f"Copying {datalake_src} to {datalake_dst}")
                    shutil.copy2(datalake_src, datalake_dst)
                    
                    if verify_file_integrity(datalake_src, datalake_dst):
                        logging.info(f"Successfully verified: {datalake_dst}")
                    else:
                        logging.error(f"File verification failed: {datalake_dst}")
                        os.remove(datalake_dst)
    
    # Process variant embeddings (p-row, p-col, p-both) - only datalake
    variants = ['p-row', 'p-col', 'p-both']
    for variant in variants:
        variant_src_base = Path(f"data/{benchmark}-{variant}/vectors")
        if not variant_src_base.exists():
            logging.warning(f"Variant directory not found: {variant_src_base}")
            continue
        
        # Process only datalake embeddings for each model type
        for model_type in model_types:
            # Handle datalake embeddings
            datalake_src = variant_src_base / f"datalake_vectors_{model_type}.pkl"
            if datalake_src.exists():
                datalake_dst = benchmark_dir / f"hytrel_datalake_embeddings_{model_type}_{variant}.pkl"
                if not datalake_dst.exists():
                    logging.info(f"Copying {datalake_src} to {datalake_dst}")
                    shutil.copy2(datalake_src, datalake_dst)
                    
                    if verify_file_integrity(datalake_src, datalake_dst):
                        logging.info(f"Successfully verified: {datalake_dst}")
                    else:
                        logging.error(f"File verification failed: {datalake_dst}")
                        os.remove(datalake_dst)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reorganize embeddings into a centralized structure')
    parser.add_argument('benchmark', type=str, help='Benchmark name (e.g., santos, tus, tusLarge)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing files')
    
    args = parser.parse_args()
    setup_logging()
    reorganize_embeddings(args.benchmark)

if __name__ == '__main__':
    main()