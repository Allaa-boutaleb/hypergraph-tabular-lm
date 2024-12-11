import os
import torch
import pickle
import argparse
import pandas as pd
import json
import time
import warnings
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger
import re
from typing import List, Tuple

# Ignore warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoConfig
from model import Encoder
from data import BipartiteData, CAP_TAG, HEADER_TAG, ROW_TAG, MISSING_CAP_TAG, MISSING_CELL_TAG, MISSING_HEADER_TAG
from parallel_clean import clean_cell_value
from torch_geometric.data import Batch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from run_pretrain import OptimizerConfig, DataArguments


def extract_vectors(model, input_data):
    """Extract embeddings from the model"""
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        embeddings = model(input_data[0])  # Note: input_data[0] as per compute_embeddings.py
    duration = time.time() - start_time
    return embeddings, duration


def remove_special_characters(text):
    """Remove special characters from text"""
    return ''.join(char for char in text if ord(char) != 0x7f)

def get_model():
    """Initialize the model with proper configuration"""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    tokenizer.add_tokens(new_tokens)
    
    # Initialize model config
    model_config = AutoConfig.from_pretrained('bert-base-uncased')
    model_config.update({
        'vocab_size': len(tokenizer),
        "pre_norm": False,
        "activation_dropout": 0.1,
        "gated_proj": False,
        "contrast_bipartite_edge": True  # Needed for contrast models
    })
    
    # Initialize model
    encoder_model = Encoder(model_config)
    return encoder_model, tokenizer

class EmbeddingGenerator:
    def __init__(self, checkpoint_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
        self.tokenizer.add_tokens(new_tokens)
        
        # Initialize model config
        model_config = AutoConfig.from_pretrained('bert-base-uncased')
        model_config.update({
            'vocab_size': len(self.tokenizer),
            "pre_norm": False,
            "activation_dropout": 0.1,
            "gated_proj": False,
            "contrast_bipartite_edge": True
        })
        
        # Initialize model
        self.model = Encoder(model_config)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()

 

def get_benchmark_variants(benchmark_name):
    """Get all variants of a benchmark if they exist"""
    variants = [benchmark_name]  # Always include the original benchmark
    
    # Check for potential variants
    variant_suffixes = ['-p-col']

    base_path = 'data'
    
    for suffix in variant_suffixes:
        variant = f"{benchmark_name}{suffix}"
        variant_path = os.path.join(base_path, variant)
        if os.path.exists(variant_path):
            variants.append(variant)
    
    return variants

class EmbeddingGenerator:
    def __init__(self, checkpoint_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
        self.tokenizer.add_tokens(new_tokens)
        
        # Initialize model config
        model_config = AutoConfig.from_pretrained('bert-base-uncased')
        model_config.update({
            'vocab_size': len(self.tokenizer),
            "pre_norm": False,
            "activation_dropout": 0.1,
            "gated_proj": False,
            "contrast_bipartite_edge": True
        })
        
        # Initialize model
        self.model = Encoder(model_config)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_dir):
        """Load DeepSpeed checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        try:
            state_dict = torch.load(checkpoint_dir)
            new_state_dict = OrderedDict()

            # Handle LoRA checkpoint
            if any('base_layer' in k for k in state_dict.keys()):
                logger.info("Detected LoRA checkpoint, merging weights...")
                
                # First, handle all non-LoRA parameters
                for k, v in state_dict.items():
                    if 'base_layer' not in k and 'lora' not in k:
                        # Remove any module prefix if present
                        if k.startswith('module.model.'):
                            name = k[13:]
                        elif k.startswith('model.'):
                            name = k[6:]
                        else:
                            name = k
                        new_state_dict[name] = v

                # Then handle LoRA parameters
                for k, v in state_dict.items():
                    if 'base_layer' in k:
                        # Extract the base name by removing '.base_layer'
                        base_name = k.replace('.base_layer', '')
                        if base_name.startswith('module.model.'):
                            base_name = base_name[13:]
                        elif base_name.startswith('model.'):
                            base_name = base_name[6:]
                        
                        # Store base weights
                        if base_name not in new_state_dict:
                            new_state_dict[base_name] = v
                        
                        # Get LoRA parameters
                        lora_a_key = k.replace('base_layer', 'lora.lora_A')
                        lora_b_key = k.replace('base_layer', 'lora.lora_B')
                        lora_a = state_dict.get(lora_a_key)
                        lora_b = state_dict.get(lora_b_key)
                        
                        if lora_a is not None and lora_b is not None:
                            # Merge LoRA weights with base weights
                            try:
                                merged_weight = v + torch.mm(lora_a, lora_b).reshape(v.shape)
                                new_state_dict[base_name] = merged_weight
                            except Exception as e:
                                logger.warning(f"Failed to merge LoRA weights for {base_name}: {str(e)}")
                                # Keep the base weights if merging fails
                                new_state_dict[base_name] = v
                
                logger.info("Successfully merged LoRA weights")
            
            # Handle regular checkpoint
            else:
                try:
                    # Try original DeepSpeed format
                    for k, v in state_dict['module'].items():
                        if 'model' in k:
                            name = k[13:]  # remove `module.model.`
                            new_state_dict[name] = v
                    logger.info("Loaded DeepSpeed checkpoint")
                    
                except KeyError:
                    # Try new format
                    for k, v in state_dict.items():
                        if k.startswith('module.model.'):
                            name = k[13:]  # remove `module.model.`
                        elif k.startswith('model.'):
                            name = k[6:]  # remove `model.`
                        else:
                            name = k
                        new_state_dict[name] = v
                    logger.info("Loaded regular checkpoint")
                    
            # Load the state dict with strict=False to allow missing keys
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            logger.success("Successfully loaded checkpoint")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def _tokenize_word(self, word: str, tokenizer: AutoTokenizer) -> Tuple[List[str], List[int]]:
        """Tokenize a word using HyTrel's method with scientific notation handling"""
        number_pattern = re.compile(r"(\d+)\.?(\d*)")

        def number_repl(matchobj):
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                exponent = len(pre) - 1
            else:
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return f"{pre}{post.rstrip('0')} scinotexp {exponent}"

        def apply_scientific_notation(line):
            return re.sub(number_pattern, number_repl, line)

        word = apply_scientific_notation(word)
        wordpieces = tokenizer.tokenize(word)[:128]  # Using 128 instead of 64
        mask = [1] * len(wordpieces) + [0] * (128 - len(wordpieces))
        wordpieces += ['[PAD]'] * (128 - len(wordpieces))
        return wordpieces, mask

    def _table2graph_columns_only(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_rows=100, max_cols=100) -> BipartiteData:
        """Create graph matching HyTrel's structure with improved efficiency"""
        # Pre-create padding sequences
        max_token_length = 128
        pad_sequence = ['[PAD]'] * max_token_length
        default_mask = [0] * max_token_length
        
        # Create default sequences using the proper missing tags
        default_cell_pieces = [MISSING_CELL_TAG] + pad_sequence[1:]
        default_cell_mask = [1] + default_mask[1:]
        default_row_pieces = ['[ROW]'] + pad_sequence[1:]
        default_row_mask = [1] + default_mask[1:]
        default_tab_pieces = [MISSING_CAP_TAG] + pad_sequence[1:]
        default_tab_mask = [1] + default_mask[1:]
        default_head_pieces = [MISSING_HEADER_TAG] + pad_sequence[1:]
        default_head_mask = [1] + default_mask[1:]

        # Limit dimensions and validate
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        
        if df.empty or len(df.columns) == 0:
            logger.warning("Skipping empty table")
            return None
        
        header = [str(col).strip() for col in df.columns]
        
        # Initialize containers
        wordpieces_xs_all, mask_xs_all = [], []
        wordpieces_xt_all, mask_xt_all = [], []
        nodes, edge_index = [], []
        
        # Process table token
        wordpieces_xt_all.append(default_tab_pieces)
        mask_xt_all.append(default_tab_mask)
        
        # Process headers consistently
        for head in header:
            if not head or pd.isna(head):
                wordpieces_xt_all.append(default_head_pieces)
                mask_xt_all.append(default_head_mask)
            else:
                wordpieces, mask = self._tokenize_word(head, tokenizer)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
        
        # Add row tokens
        for _ in range(min(len(df), max_rows)):
            wordpieces_xt_all.append(default_row_pieces)
            mask_xt_all.append(default_row_mask)
        
        # Process cells efficiently
        for row_i, row in enumerate(df.itertuples()):
            if row_i >= max_rows:
                break
            for col_i, cell in enumerate(row[1:]):
                if col_i >= max_cols:
                    break
                
                if pd.isna(cell):
                    wordpieces_xs_all.append(default_cell_pieces)
                    mask_xs_all.append(default_cell_mask)
                else:
                    word = remove_special_characters(' '.join(str(cell).split()[:max_token_length]))
                    wordpieces, mask = self._tokenize_word(word, tokenizer)
                    wordpieces_xs_all.append(wordpieces)
                    mask_xs_all.append(mask)
                
                node_id = len(nodes)
                nodes.append(node_id)
                edge_index.extend([
                    [node_id, 0],  # table-level
                    [node_id, col_i + 1],  # column-level
                    [node_id, row_i + len(header) + 1]  # row-level
                ])
        
        # Convert to tensors efficiently
        xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
        xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        
        # Add column mask for proper column identification
        col_mask = torch.zeros(len(wordpieces_xt_all), dtype=torch.long)
        col_mask[1:len(df.columns)+1] = 1  # Mark column hyperedges
        
        return BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, col_mask=col_mask)

    def extract_columns(self, outputs: torch.Tensor, num_cols: int) -> torch.Tensor:
        """Extract column embeddings from the model output"""
        # Get hyperedge embeddings and extract columns (skip table token)
        return outputs[1][1:num_cols+1]

    def process_directory(self, csv_dir, output_vectors_path, batch_size=4, max_rows=100, max_cols=100):
        """Process all CSV files in a directory and generate embeddings"""
        logger.info(f"Processing directory: {csv_dir}")
        logger.info(f"Using max_rows={max_rows}, max_cols={max_cols}")
        os.makedirs(os.path.dirname(output_vectors_path), exist_ok=True)
        
        data_embeds = []
        inference_time = 0
        total_time = 0
        skipped_files = 0
        
        total_start = time.time()
        csv_files = list(Path(csv_dir).glob('*.csv'))
        
        for csv_file in tqdm(csv_files, desc=f"Processing {Path(csv_dir).name}"):
            try:
                # Read and validate CSV
                df = pd.read_csv(csv_file)
                if len(df) == 0:
                    skipped_files += 1
                    continue
                
                # Convert table to graph
                graph = self._table2graph_columns_only(df, self.tokenizer, max_rows=max_rows, max_cols=max_cols)
                if graph is None:
                    skipped_files += 1
                    continue
                
                # Move graph to device
                graph = graph.to(self.device)
                
                # Generate embeddings
                inference_start = time.time()
                with torch.no_grad():
                    outputs = self.model(graph)
                inference_time += time.time() - inference_start
                
                # Extract column embeddings
                num_cols = len(df.columns) if len(df.columns) <= max_cols else max_cols
                col_embeddings = self.extract_columns(outputs, num_cols)
                
                # Store results
                data_embeds.append((csv_file.stem, col_embeddings.cpu().numpy()))
                
                # Clean up
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                skipped_files += 1
                continue
        
        total_time = time.time() - total_start
        
        # Save embeddings
        logger.info(f"Saving embeddings to {output_vectors_path}")
        with open(output_vectors_path, 'wb') as f:
            pickle.dump(data_embeds, f)
        
        # Return statistics
        stats = {
            "total_time": total_time,
            "inference_time": inference_time,
            "processed_tables": len(data_embeds),
            "skipped_files": skipped_files
        }
        
        return stats


def process_benchmark(benchmark, generator, args, variant):
    """Process a single benchmark or variant"""
    logger.info(f"Processing benchmark: {benchmark}")
    
    # Setup paths
    dataset_path = os.path.join('data', benchmark)
    datalake_path = os.path.join(dataset_path, 'datalake')
    query_path = os.path.join(dataset_path, 'query')
    vectors_path = os.path.join(dataset_path, 'vectors')
    os.makedirs(vectors_path, exist_ok=True)

    # Prepare file names with variant
    variant_suffix = f"_{variant}"
    datalake_vectors_path = os.path.join(vectors_path, f'datalake_vectors{variant_suffix}.pkl')
    query_vectors_path = os.path.join(vectors_path, f'query_vectors{variant_suffix}.pkl')
    timing_path = os.path.join(vectors_path, f'timing_stats{variant_suffix}.json')

    # Initialize timing statistics
    timing_stats = {
        "benchmark": benchmark,
        "variant": variant,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint_dir,
    }

    # Process directories
    logger.info(f"Processing datalake directory for {benchmark}...")
    datalake_stats = generator.process_directory(datalake_path, datalake_vectors_path, 
                                               max_rows=args.max_rows, max_cols=args.max_cols)
    
    logger.info(f"Processing query directory for {benchmark}...")
    query_stats = generator.process_directory(query_path, query_vectors_path, 
                                            max_rows=args.max_rows, max_cols=args.max_cols)
    
    # Update timing statistics
    timing_stats.update({
        "datalake": datalake_stats,
        "query": query_stats
    })
    
    # Save timing statistics
    logger.info(f"Saving timing statistics to {timing_path}")
    with open(timing_path, 'w') as f:
        json.dump(timing_stats, f, indent=4)
    
    return timing_stats

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings for table columns using HyTrel')
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--max_rows', type=int, default=100)
    parser.add_argument('--max_cols', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # Extract variant from checkpoint_dir
    variant_path = args.checkpoint_dir.split('hypergraph-tabular-lm/checkpoints/')[-1]
    variant = variant_path.split('/')[0]  # Get the directory name after checkpoints/
    variant = variant.split('_')[-1]      # Get the last part after splitting by underscore
    logger.info(f"Extracted variant: {variant}")

    # Get all variants of the benchmark
    benchmarks_to_process = get_benchmark_variants(args.benchmark)
    logger.info(f"Found benchmarks to process: {benchmarks_to_process}")
    
    # Initialize model once and reuse for all variants
    generator = EmbeddingGenerator(args.checkpoint_dir)
    
    all_stats = {}
    for benchmark in benchmarks_to_process:
        try:
            stats = process_benchmark(benchmark, generator, args, variant)
            all_stats[benchmark] = stats
            logger.success(f"Successfully processed benchmark: {benchmark}")
        except Exception as e:
            logger.error(f"Error processing benchmark {benchmark}: {str(e)}")
            continue
    
    # Save combined statistics
    combined_stats_path = os.path.join('data', args.benchmark, 'vectors', f'combined_stats_{variant}.json')
    with open(combined_stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    logger.success("All benchmark processing completed successfully!")

if __name__ == '__main__':
    main()