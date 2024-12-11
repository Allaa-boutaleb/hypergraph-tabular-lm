import os
import json
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from checkPrecisionRecall import calcMetrics, loadDictionaryFromPickleFile
from naive_search import NaiveSearcher
from tqdm import tqdm

def setup_directories(benchmark):
    """Create output directory structure for the benchmark"""
    base_dir = Path(f"output/{benchmark}")
    variants = ['original', 'p-row', 'p-col', 'p-both']
    model_types = ['finetuned', 'lora', 'scratch', 'pretrained']
    
    for variant in variants:
        for model_type in model_types:
            (base_dir / variant / model_type).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_embeddings(benchmark, variant):
    """Load embeddings for a specific benchmark variant"""
    base_path = Path(f"vectors/{benchmark}")
    model_types = ['finetuned', 'lora', 'scratch', 'pretrained']
    
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

def calculate_similarity_metrics(original_embeddings, variant_embeddings):
    """Calculate similarity metrics between original and variant embeddings"""
    euclidean_distances = []
    cosine_similarities = []
    
    # Convert lists of tuples to dictionaries for easier lookup
    orig_dict = {entry[0]: entry[1] for entry in original_embeddings}
    var_dict = {entry[0]: entry[1] for entry in variant_embeddings}
    
    # Compare embeddings for matching table IDs
    for table_id in orig_dict:
        if table_id in var_dict:
            # Take mean of embeddings to get table-level vector
            orig = np.mean(orig_dict[table_id], axis=0)  # Convert matrix to vector
            var = np.mean(var_dict[table_id], axis=0)    # Convert matrix to vector
            
            euclidean_distances.append(euclidean(orig, var))
            cosine_similarities.append(1 - cosine(orig, var))
    
    return {
        'euclidean': {
            'mean': float(np.mean(euclidean_distances)),
            'std': float(np.std(euclidean_distances)),
            'min': float(np.min(euclidean_distances)),
            'max': float(np.max(euclidean_distances))
        },
        'cosine': {
            'mean': float(np.mean(cosine_similarities)),
            'std': float(np.std(cosine_similarities)),
            'min': float(np.min(cosine_similarities)),
            'max': float(np.max(cosine_similarities))
        }
    }, euclidean_distances, cosine_similarities

def evaluate_benchmark(benchmark_name, model_type):
    """Main evaluation function for a benchmark and model type"""
    # Parameters for different benchmarks
    params = {
        'santos': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0
        },
        'pylon': {
            'max_k': 10, 
            'k_range': 1, 
            'sample_size': None, 
            'threshold': 0.1,
            'scale': 1.0
        },
        'tus': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 150, 
            'threshold': 0.1,
            'scale': 1.0
        },
        'tusLarge': {
            'max_k': 60, 
            'k_range': 10, 
            'sample_size': 100, 
            'threshold': 0.1,
            'scale': 1.0
        }
    }[benchmark_name]
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation for {benchmark_name} - {model_type}")
    print(f"{'='*80}")
    
    output_dir = setup_directories(benchmark_name, model_type)
    base_path = Path(f"vectors/{benchmark_name}")

    # Load ground truth 
    gt_path = base_path / "benchmark.pkl"
    
    variants = ['original', 'p-row', 'p-col', 'p-both']
    results = {}
    
    # Add debug: Load and print first few entries from ground truth
    print("\nDEBUG: Loading ground truth file...")
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    print("\nFirst few entries in ground truth:")
    for i, (query, candidates) in enumerate(gt_data.items()):
        if i < 3:  # Show first 3 entries
            print(f"\nQuery: {query}")
            print(f"Candidates: {candidates[:5]}...")  # Show first 5 candidates
    
    for variant in variants:
        print(f"\n{'='*50}")
        print(f"Processing {model_type} - {variant} variant")
        print(f"{'='*50}")
        
        queries, datalake = load_embeddings(benchmark_name, model_type, variant)
        if queries is None or datalake is None:
            print(f"Skipping {variant}: embeddings not found")
            continue
        
        # Debug: Print first few queries and their embeddings
        print("\nDEBUG: First few queries from embeddings:")
        query_items = queries.items() if isinstance(queries, dict) else queries
        for i, query in enumerate(query_items):
            if i < 3:  # Show first 3 queries
                print(f"\nQuery ID: {query[0]}")
                print(f"Query embedding shape: {query[1].shape}")
        
        # Create temporary pickle files for the NaiveSearcher
        temp_datalake_path = f"temp_{model_type}_{variant}_datalake.pkl"
        with open(temp_datalake_path, 'wb') as f:
            pickle.dump(datalake, f)
        
        # Do the search
        searcher = NaiveSearcher(temp_datalake_path, scale=params['scale'])
        returnedResults = {}
        
        # Debug: Process just the first query and print detailed results
        first_query = next(iter(query_items))
        print("\nDEBUG: Processing first query in detail:")
        print(f"Query ID: {first_query[0]}")
        
        search_results = searcher.topk(
            enc='cl',
            query=first_query,
            K=params['max_k'],
            threshold=params['threshold']
        )
        
        # Debug: Print search results for first query
        print("\nSearch results for first query:")
        for i, result in enumerate(search_results):
            print(f"Result {i+1}: {result}")  # This should show (score, table_id)
        
        # Debug: Check if this query exists in ground truth
        query_id = first_query[0] if first_query[0].endswith('.csv') else f"{first_query[0]}.csv"
        print(f"\nLooking for query {query_id} in ground truth...")
        if query_id in gt_data:
            print(f"Found in ground truth! First few expected matches: {gt_data[query_id][:5]}")
            print("\nComparing returned results with ground truth:")
            returned_set = set(r[1] for r in search_results)
            gt_set = set(gt_data[query_id])
            correct_matches = returned_set.intersection(gt_set)
            print(f"Number of correct matches in top {params['max_k']}: {len(correct_matches)}")
            if correct_matches:
                print(f"Correct matches: {correct_matches}")
        else:
            print("NOT found in ground truth!")
            print(f"Available ground truth keys (first 5): {list(gt_data.keys())[:5]}")
        
        # Debug: Print some example key formats
        print("\nExample formats:")
        print(f"Query ID format: {query_id}")
        print(f"First result format: {search_results[0][1] if search_results else 'No results'}")
        print(f"Ground truth key format example: {next(iter(gt_data.keys()))}")
        
        # Now continue with normal processing
        returnedResults[query_id] = [
            f"{r[1]}.csv" if not r[1].endswith('.csv') else r[1] 
            for r in search_results
        ]
        
        print("\nFirst entry in returnedResults:")
        print(f"Query: {query_id}")
        print(f"Results: {returnedResults[query_id][:5]}...")
        
        # Stop after processing the first query in the first variant
        if variant == 'original':
            print("\nStopping after first query for debugging. Comment out this section to process all queries.")
            break
        
        # Create temporary pickle files for the NaiveSearcher
        temp_datalake_path = f"temp_{model_type}_{variant}_datalake.pkl"
        with open(temp_datalake_path, 'wb') as f:
            pickle.dump(datalake, f)
        
        # Do the search
        searcher = NaiveSearcher(temp_datalake_path, scale=params['scale'])
        returnedResults = {}
        
        # Sample queries if needed
        if isinstance(queries, list):
            queries.sort(key=lambda x: x[0])
        if params['sample_size'] is not None:
            np.random.seed(42)
            if isinstance(queries, list):
                indices = np.random.choice(len(queries), size=params['sample_size'], replace=False)
                queries = [queries[i] for i in indices]
            else:
                # Handle dictionary case
                query_items = list(queries.items())
                indices = np.random.choice(len(query_items), size=params['sample_size'], replace=False)
                queries = dict([query_items[i] for i in indices])
        
        # Process queries
        # Process queries
        query_items = queries.items() if isinstance(queries, dict) else queries
        for query in tqdm(query_items, desc="Processing queries"):
            search_results = searcher.topk(
                enc='cl',
                query=query,
                K=params['max_k'],
                threshold=params['threshold']
            )
            # Hot fix: Append .csv to both query ID and result IDs since hytrel doesn't do it
            query_id = query[0] if query[0].endswith('.csv') else f"{query[0]}.csv"
            returnedResults[query_id] = [
                f"{r[1]}.csv" if not r[1].endswith('.csv') else r[1] 
                for r in search_results
            ]
        
        # Clean up temporary file
        os.remove(temp_datalake_path)
        
        # Calculate metrics
        metrics = calcMetrics(
            max_k=params['max_k'],
            k_range=params['k_range'],
            resultFile=returnedResults,
            gtPath=gt_path,
            record=False,
            verbose=False
        )
        
        # Save metrics
        with open(output_dir / variant / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results[variant] = metrics
        
        # Generate plots
        plot_variant_metrics(metrics, output_dir / variant / 'figures', variant)
        
        # Calculate similarity metrics if not original
        if variant != 'original':
            original_datalake = load_embeddings(benchmark_name, model_type, 'original')[1]
            similarity_metrics, euclidean_distances, cosine_similarities = calculate_similarity_metrics(
                original_datalake, datalake)
            
            with open(output_dir / variant / 'similarity_metrics.json', 'w') as f:
                json.dump(similarity_metrics, f, indent=2)
            
            raw_distances = {
                'euclidean': euclidean_distances,
                'cosine': cosine_similarities
            }
            with open(output_dir / variant / 'raw_distances.json', 'w') as f:
                json.dump(raw_distances, f)
            
            plot_similarity_metrics(similarity_metrics, output_dir / variant / 'figures', variant)
    
    # Generate combined comparison plots
    plot_comparison_metrics(results, output_dir)


def plot_variant_metrics(metrics, output_dir, variant):
    """Generate plots for individual variant metrics showing P@k, R@k, MAP@k, and F1@k"""
    plt.figure(figsize=(12, 8))
    
    # Extract k values and metrics
    k_values = metrics['used_k']
    
    # Extract different metrics for each k
    precision = [metrics['metrics_at_k'][k]['precision'] for k in k_values]
    recall = [metrics['metrics_at_k'][k]['recall'] for k in k_values]
    map_scores = [metrics['metrics_at_k'][k]['map'] for k in k_values]
    f1_scores = [metrics['metrics_at_k'][k]['f1'] for k in k_values]
    
    # Plot all metrics
    plt.plot(k_values, precision, 'b-o', label='Precision@k')
    plt.plot(k_values, recall, 'r-s', label='Recall@k')
    plt.plot(k_values, map_scores, 'g-^', label='MAP@k')
    plt.plot(k_values, f1_scores, 'y-d', label='F1@k')
    
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.title(f'Retrieval Metrics for {variant}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_dir / 'retrieval_metrics.pdf')
    plt.close()

def plot_similarity_metrics(metrics, output_dir, variant):
    """Generate plots for similarity metrics between original and variant embeddings"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Euclidean distance distribution
    euclidean_stats = metrics['euclidean']
    ax1.bar(['Mean', 'Std', 'Min', 'Max'], 
            [euclidean_stats['mean'], euclidean_stats['std'], 
             euclidean_stats['min'], euclidean_stats['max']], 
            color='blue', alpha=0.6)
    ax1.set_title(f'Euclidean Distance\n{variant} vs Original')
    ax1.grid(True)
    
    # Plot Cosine similarity distribution
    cosine_stats = metrics['cosine']
    ax2.bar(['Mean', 'Std', 'Min', 'Max'], 
            [cosine_stats['mean'], cosine_stats['std'], 
             cosine_stats['min'], cosine_stats['max']], 
            color='green', alpha=0.6)
    ax2.set_title(f'Cosine Similarity\n{variant} vs Original')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_metrics.pdf')
    plt.close()

def plot_comparison_metrics(results, output_dir):
    """Generate final comparison plots across all variants"""
    variants = [v for v in results.keys() if v != 'original']
    
    # Create figures directory if it doesn't exist
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Collect all distances and similarities for box plots
    euclidean_data = []
    cosine_data = []
    euclidean_means = []
    cosine_means = []
    
    for variant in variants:
        with open(output_dir / variant / 'similarity_metrics.json', 'r') as f:
            metrics = json.load(f)
            # Store means for the bar plot
            euclidean_means.append(metrics['euclidean']['mean'])
            cosine_means.append(metrics['cosine']['mean'])
            
            # Read raw distances from files for box plots
            with open(output_dir / variant / 'raw_distances.json', 'r') as f:
                raw_data = json.load(f)
                euclidean_data.append(raw_data['euclidean'])
                cosine_data.append(raw_data['cosine'])
    
    # Create distance/similarity plot with box plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Box plots for Euclidean distances
    bp1 = ax1.boxplot(euclidean_data, labels=variants, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    ax1.bar(range(1, len(variants) + 1), euclidean_means, alpha=0.3, color='blue', width=0.3)
    ax1.set_title('Euclidean Distance Distribution (Original vs Variants)')
    ax1.set_ylabel('Distance\n(higher = more different)')
    ax1.grid(True, alpha=0.3)
    
    # Box plots for Cosine similarities
    bp2 = ax2.boxplot(cosine_data, labels=variants, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    ax2.bar(range(1, len(variants) + 1), cosine_means, alpha=0.3, color='green', width=0.3)
    ax2.set_title('Cosine Similarity Distribution (Original vs Variants)')
    ax2.set_ylabel('Similarity\n(lower = more different)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'variant_distances.pdf')
    plt.close()
    
    # Performance Impact Plot
    plt.figure(figsize=(12, 6))
    metrics_to_compare = ['precision', 'recall', 'map', 'f1']
    x = np.arange(len(variants))
    bar_width = 0.2
    
    for i, metric in enumerate(metrics_to_compare):
        original_score = results['original'][metric][9]  # k=10 is at index 9
        # Calculate absolute performance drop (original - variant)
        performance_drops = [
            original_score - results[variant][metric][9]
            for variant in variants
        ]
        
        plt.bar(x + i * bar_width, performance_drops, bar_width, 
                label=f'{metric.upper()}@10')
    
    plt.xlabel('Variants')
    plt.ylabel('Performance Drop\n(higher = worse)')
    plt.title('Performance Impact vs Original')
    plt.xticks(x + bar_width * (len(metrics_to_compare) - 1) / 2, variants)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'performance_delta.pdf')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=['santos', 'tus', 'tusLarge', 'pylon'],
                       help="Benchmark to evaluate")
    parser.add_argument("model_type", choices=['pretrained', 'lora', 'scratch', 'finetuned'],
                       help="Model type to evaluate")
    args = parser.parse_args()
    
    evaluate_benchmark(args.benchmark, args.model_type)