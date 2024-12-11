import os
import json
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine, euclidean
from checkPrecisionRecall import calcMetrics, loadDictionaryFromPickleFile
from naive_search import NaiveSearcher
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def setup_directories(benchmark):
    """Create output directory structure for the benchmark"""
    base_dir = Path(f"output/{benchmark}")
    variants = ['original', 'p-col']
    model_types = ['pretrained', 'finetuned', 'scratch']
    
    for variant in variants:
        for model_type in model_types:
            (base_dir / variant / model_type).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_embeddings(benchmark, variant):
    """Load embeddings for a specific benchmark variant"""
    base_path = Path(f"vectors/{benchmark}")
    model_types = ['pretrained', 'scratch', 'finetuned']
    
    try:
        # For query embeddings, we use the non-permuted version
        query_embeddings = {}
        for model_type in model_types:
            query_path = base_path / f"hytrel_query_embeddings_{model_type}.pkl"
            if query_path.exists():
                queries = loadDictionaryFromPickleFile(query_path)
                query_embeddings[model_type] = queries
        
        # For datalake embeddings, we use the variant-specific version
        datalake_embeddings = {}
        variant_suffix = f"_{variant}" if variant else ""
        for model_type in model_types:
            datalake_path = base_path / f"hytrel_datalake_embeddings_{model_type}{variant_suffix}.pkl"
            if datalake_path.exists():
                datalake = loadDictionaryFromPickleFile(datalake_path)
                datalake_embeddings[model_type] = datalake
        
        return query_embeddings, datalake_embeddings
    except FileNotFoundError as e:
        print(f"Warning: Could not load embeddings for {benchmark}/{variant}: {e}")
        return None, None

def load_table_structure(table_path):
    """Load CSV table to get column names and order"""
    try:
        df = pd.read_csv(table_path)
        return list(df.columns)
    except:
        return None

def calculate_detailed_similarity_metrics(original_embeddings, variant_embeddings, data_dir, variant):
    """Calculate column-level similarity metrics between original and variant embeddings"""
    detailed_metrics = {"tables": []}
    benchmark = str(data_dir).split('/')[1]

    # Keep as lists to preserve order
    orig_emb_list = original_embeddings
    var_emb_list = variant_embeddings
    
    # Create mapping of table IDs to indices
    orig_indices = {x[0]: i for i,x in enumerate(orig_emb_list)}
    var_indices = {x[0]: i for i,x in enumerate(var_emb_list)}
    
    for table_id in orig_indices:
        if table_id not in var_indices:
            continue
            
            
        orig_path = Path("data") / benchmark / "datalake" / f"{table_id}.csv"
        var_path = Path("data") / f"{benchmark}-{variant}" / "datalake" / f"{table_id}.csv"
        
        orig_columns = load_table_structure(orig_path)
        var_columns = load_table_structure(var_path)
        
        if not orig_columns or not var_columns:
            continue

            
        # Get embeddings using list indices
        orig_embeddings_table = orig_emb_list[orig_indices[table_id]][1]
        var_embeddings_table = var_emb_list[var_indices[table_id]][1]
        
        if len(orig_embeddings_table) != len(var_embeddings_table):
            continue
            
        cos_sim = cosine_similarity(orig_embeddings_table, var_embeddings_table)
        
        column_similarities = []
        euclidean_distances = []
        cosine_similarities = []
        
        for col_name in orig_columns:
            if col_name in var_columns:
                orig_idx = orig_columns.index(col_name)
                var_idx = var_columns.index(col_name)
                
                orig_emb = orig_embeddings_table[orig_idx]
                var_emb = var_embeddings_table[var_idx]
                
                cos_sim_val = float(cos_sim[orig_idx][var_idx])
                euc_dist = float(np.linalg.norm(orig_emb - var_emb))
                
                
                column_similarities.append({
                    "column_name": col_name,
                    "original_position": orig_idx,
                    "permuted_position": var_idx,
                    "euclidean_distance": euc_dist,
                    "cosine_similarity": cos_sim_val
                })
                
                euclidean_distances.append(euc_dist)
                cosine_similarities.append(cos_sim_val)
        
        if column_similarities:
            table_metrics = {
                "table_name": table_id,
                "num_columns": len(orig_columns),
                "column_similarities": column_similarities,
                "aggregate_metrics": {
                    "mean_euclidean": float(np.mean(euclidean_distances)),
                    "mean_cosine": float(np.mean(cosine_similarities))
                }
            }
            detailed_metrics["tables"].append(table_metrics)
    
    return detailed_metrics

def evaluate_benchmark(benchmark_name, distances_only=False):
    """Main evaluation function for a benchmark"""
    # Parameters from run_tus_all.py and test_naive_search.py
    params = {
        'santos': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'pylon': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tus': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 150, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        },
        'tusLarge': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 100, 
            'threshold': 0.1,
            'scale': 1.0,
            'encoder': 'cl',
            'matching': 'exact',
            'table_order': 'column'
        }
    }[benchmark_name]
    
    output_dir = setup_directories(benchmark_name)
    base_path = Path(f"vectors/{benchmark_name}")
    data_path = Path(f"data/{benchmark_name}")
    gt_path = base_path / "benchmark.pkl"

    # model_types = ['pretrained', 'finetuned', 'lora', 'scratch']
    model_types = ['pretrained', 'finetuned', 'scratch']

    
    if distances_only:
        variants_dist = ['p-col']
        for model_type in model_types:
            print(f"\nProcessing distance metrics for model: {model_type}")
            for variant in variants_dist:
                try:
                    original_path = base_path / f"hytrel_datalake_embeddings_{model_type}.pkl"
                    variant_path = base_path / f"hytrel_datalake_embeddings_{model_type}_{variant}.pkl"
                    
                    if not original_path.exists() or not variant_path.exists():
                        continue
                    
                    original_datalake = loadDictionaryFromPickleFile(original_path)
                    variant_datalake = loadDictionaryFromPickleFile(variant_path)
                    
                    detailed_metrics = calculate_detailed_similarity_metrics(
                        original_datalake, 
                        variant_datalake,
                        data_path,
                        variant
                    )
                    
                    # Save detailed metrics
                    with open(output_dir / variant / model_type / 'raw_distances.json', 'w') as f:
                        json.dump(detailed_metrics, f, indent=2)
                        
                    print(f"Updated detailed metrics for {variant}/{model_type}")
                    
                except Exception as e:
                    print(f"Error processing {variant}/{model_type}: {e}")
        return

    results = {}
    
    for model_type in model_types:
        print(f"\nProcessing model: {model_type}")
        
        # Load query embeddings once for this model type
        query_path = base_path / f"hytrel_query_embeddings_{model_type}.pkl"
        if not query_path.exists():
            print(f"Skipping {model_type}: query embeddings not found")
            continue
            
        queries = loadDictionaryFromPickleFile(query_path)
        queries.sort(key=lambda x: x[0])
        
        # Sample queries for tus and tusLarge
        if params['sample_size'] is not None:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(queries), size=params['sample_size'], replace=False)
            queries = [queries[i] for i in indices]
        
        variants = ['original']

        for variant in variants:
            variant_suffix = f"_{variant}" if variant != 'original' else ""
            datalake_path = base_path / f"hytrel_datalake_embeddings_{model_type}{variant_suffix}.pkl"
            
            if not datalake_path.exists():
                print(f"Skipping {variant}/{model_type}: datalake embeddings not found")
                continue
            
            print(f"Processing variant: {variant}")
            
            # Do the search
            searcher = NaiveSearcher(str(datalake_path), scale=params['scale'])
            returnedResults = {}
            
            for query in tqdm(queries, desc="Processing queries", unit="query"):
                search_results = searcher.topk(
                    enc=params['encoder'],
                    query=query,
                    K=params['max_k'], 
                    threshold=params['threshold']
                )
                # Append .csv to table names to match benchmark format
                returnedResults[query[0] + '.csv'] = [r[1] + '.csv' for r in search_results]
            
            # Calculate metrics
            metrics = calcMetrics(
                max_k=params['max_k'],
                k_range=params['k_range'],
                resultFile=returnedResults,
                gtPath=gt_path,
                record=False,
                verbose=False
            )
            
            # Save detailed metrics
            with open(output_dir / variant / model_type / 'detailed_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            results[f"{variant}_{model_type}"] = metrics['system_metrics']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate benchmark and calculate similarity metrics")
    parser.add_argument("benchmark", 
                       choices=['santos', 'tus', 'tusLarge', 'pylon'],
                       help="Benchmark to evaluate")
    parser.add_argument("--distances_only", 
                       action="store_true",
                       help="Only recalculate detailed distance metrics without redoing evaluation")
    parser.add_argument("--data_dir",
                       type=str,
                       default=None,
                       help="Optional: Override default data directory path")
    
    args = parser.parse_args()
    
    # Update data path if provided
    if args.data_dir:
        data_path = Path(args.data_dir)
    
    evaluate_benchmark(args.benchmark, args.distances_only)