
from logging import WARN
from re import M
from urllib import parse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import yaml
import json
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at .* were not used when initializing .*")
import gc
from datasets import Dataset
from datetime import datetime
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
import zlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from rouge_score import rouge_scorer
import hdbscan
from mpl_toolkits.mplot3d import Axes3D
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import torch.nn.functional as F
import zlib
import math
import argparse
import time
import pickle

from tkinter import NO
import wandb
import joblib

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM")

# Add argument parser
parser = argparse.ArgumentParser(description='Run experiments with configurable parameters')
parser.add_argument('--n_tokens', type=float, default=0.3, help='Number of tokens for strategy')
parser.add_argument('--test_size', type=float, default=0.2, help='Size of test subset')
parser.add_argument('--non_debug', default=False, action='store_true', help='Disable debug mode')
parser.add_argument('--rephrased_original', default=False, action='store_true', help='Use rephrased original text')
parser.add_argument('--max_neighbors', type=int, default=15, help='Maximum neighbors for strategy')
parser.add_argument('--neighbor_dist', type=int, default=20, help='Neighbor distance for strategy')
parser.add_argument('--wb_name', type=str, default=None, help='Weights & Biases run name')
parser.add_argument('--subset_size', type=int, default=1000, help='Subset size for experiments')
parser.add_argument('--wandb_mode', type=str, default='online', help='Weights & Biases mode (online, offline, disabled)')
parser.add_argument('--compute_baselines', default=False, action='store_true', help='Whether to compute baseline results')
parser.add_argument('--train_size_test', default=False, action='store_true', help='Enable learning curve test')
parser.add_argument('--train_size', type=int, default=1000, help='Training set size for learning curve')
parser.add_argument('--n_repeats', type=int, default=3, help='Number of repetitions for averaging learning curve results')
parser.add_argument('--max_prompt_tokens', type=int, default=300, help='Maximum tokens for input prompt truncation')
parser.add_argument('--max_gen_tokens', type=int, default=300, help='Maximum tokens to generate as output')
parser.add_argument('--timestamp', type=str, default=None, help='Timestamp for the experiment run')
parser.add_argument('--new_classifiers', default=False, action='store_true', help='Use new classifiers for evaluation')
parser.add_argument('--transferability', default=False, action='store_true', help='Enable transferability experiments')
parser.add_argument('--save_classifiers', default=False, action='store_true', help='Save trained classifiers for future use')

args, unknown = parser.parse_known_args()

# Use the arguments
N_TOKENS = args.n_tokens
MAX_PROMPT_TOKENS = args.max_prompt_tokens
MAX_GEN_TOKENS = args.max_gen_tokens
TEST_SIZE = args.test_size
REPHRASED_ORIGINAL = args.rephrased_original
MAX_NEIGHBORS = args.max_neighbors
NEIGHBOR_DIST = args.neighbor_dist
SUBSET_SIZE = args.subset_size
WANDB_MODE = args.wandb_mode
COMPUTE_BASELINES = args.compute_baselines
DEBUG = not args.non_debug
TRAIN_SIZE_TEST = args.train_size_test
TRAIN_SIZE = args.train_size
N_REPEATS = args.n_repeats
NEW_CLASSIFIERS = args.new_classifiers
TRANSFERABILITY = args.transferability
SAVE_CLASSIFIERS = args.save_classifiers

if TRANSFERABILITY:
    SAVE_CLASSIFIERS = True
    print(f"TRANSFERABILITY enabled, setting SAVE_CLASSIFIERS to True.")

# Setup directories
curr_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
print(f"Current directory: {curr_dir}")
PROJECT_DIR = os.path.abspath(os.path.join(curr_dir, '..', '..'))
print(f"Project directory: {PROJECT_DIR}")
Unlearn_Simple_DIR = os.path.join(PROJECT_DIR, 'Unlearn-Simple')
print(f"Unlearn_Simple_DIR: {Unlearn_Simple_DIR}")

TOFU_DIR = os.path.join(Unlearn_Simple_DIR, 'TOFU')
MUSE_DIR = os.path.join(Unlearn_Simple_DIR, 'MUSE')
WMDP_DIR = os.path.join(Unlearn_Simple_DIR, 'WMDP')

# Add necessary paths
sys.path.append(PROJECT_DIR)
sys.path.append(Unlearn_Simple_DIR)
sys.path.append(os.path.join(TOFU_DIR))
sys.path.append(os.path.join(MUSE_DIR))
sys.path.append(os.path.join(WMDP_DIR))
sys.path.append(os.path.join(MUSE_DIR, 'src'))
sys.path.append(os.path.join(PROJECT_DIR, 'src'))


# COMPUTE_BASELINES = False
print(f"{MAX_GEN_TOKENS=}, {MAX_PROMPT_TOKENS=}, {TEST_SIZE=}, {COMPUTE_BASELINES=}")

STRATEGY = {
        'name': 'embeddings', 
        'peak_top_k': NEIGHBOR_DIST, 
        'n_tokens': N_TOKENS, 
        'max_neighbors': MAX_NEIGHBORS,
        'max_prompt_tokens': MAX_PROMPT_TOKENS,
        'max_gen_tokens': MAX_GEN_TOKENS,
    }

if DEBUG:
    RESULTS_DIR = os.path.join(Unlearn_Simple_DIR, 'results', 'debug', f'subset_{SUBSET_SIZE}')
    TIMESTAMP = '2025-09-17_18-58'
else:
    RESULTS_DIR = os.path.join(Unlearn_Simple_DIR, 'results', f'subset_{SUBSET_SIZE}')
    TIMESTAMP = args.timestamp if args.timestamp is not None else datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

EXPERIMENT_NAME = f'top_k_{NEIGHBOR_DIST}_k_neighbors_{MAX_NEIGHBORS}_n_tokens_{N_TOKENS}_MNT_{MAX_PROMPT_TOKENS}_test_size_{TEST_SIZE}_reph_{REPHRASED_ORIGINAL}'

# Create results storage directory
RESULTS_DIR = "comprehensive_results"
if DEBUG:
    SUBSET_SIZE = 10  # Smaller subset for debugging
    WANDB_MODE = 'disabled'  # Disable wandb in debug mode
RESULTS_DIR = os.path.join(Unlearn_Simple_DIR, RESULTS_DIR, f"subset_size-{SUBSET_SIZE}", TIMESTAMP)
os.makedirs(RESULTS_DIR, exist_ok=True)

if WANDB_MODE not in ['online', 'offline', 'disabled']:
    print(f"Invalid WANDB_MODE '{WANDB_MODE}' provided. Defaulting to 'online'.")
    WANDB_MODE = 'online'
else:
    print(f"WANDB_MODE set to '{WANDB_MODE}'.")
    

os.environ['WANDB_MODE'] = WANDB_MODE

wandb_exp_name = args.wb_name if args.wb_name is not None else f"exp_{TIMESTAMP}"
wandb.init(
    project="unlearning-evaluation",
    name=wandb_exp_name,  # Unique run name based on timestamp
    config={
        "N_TOKENS": N_TOKENS,
        "MAX_GEN_TOKENS": MAX_GEN_TOKENS,
        "MAX_PROMPT_TOKENS": MAX_PROMPT_TOKENS,
        "TEST_SIZE": TEST_SIZE,
        "SUBSET_SIZE": SUBSET_SIZE,
        "REPHRASED_ORIGINAL": REPHRASED_ORIGINAL,
        "DEBUG": DEBUG,
        'MAX_NEIGHBORS': MAX_NEIGHBORS,
        'NEIGHBOR_DIST': NEIGHBOR_DIST,
        'STRATEGY': STRATEGY,
    },
    notes="Comprehensive unlearning evaluation across models and benchmarks."
)

# Helper function for GPU memory cleanup
def cleanup_gpu_memory():
    """Enhanced GPU memory cleanup to handle fragmentation and lingering references."""
    # Force garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        # Clear cache and collect IPC
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Synchronize to ensure operations are complete
        torch.cuda.synchronize()
        
        # Reset peak memory stats (helps with monitoring and potential resets)
        torch.cuda.reset_peak_memory_stats()
        
        # Optional: Force a brief sleep to allow background cleanup (if fragmentation is an issue)
        import time
        time.sleep(0.1)  # Small delay; adjust if needed
        
        torch.cuda.ipc_collect()
        
        # Re-run empty_cache after sync (sometimes helps with stubborn allocations)
        torch.cuda.empty_cache()
        
# Clean up memory after each iteration
cleanup_gpu_memory()

print("Setup complete!")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# Import evaluation modules
import MUSE.src.eval_with_ILL as eval_with_ILL
from src.input_loss_landscape.utils import new_ILL_eval
import src.utils as project_utils

# Define all model configurations for experiments
MODEL_CONFIGS = {
    # WMDP models (LLaMA-3-8B with different unlearning methods)
    "wmdp_models": [
        "LLM-GAT/llama-3-8b-instruct-elm-checkpoint-8",
        # # "LLM-GAT/llama-3-8b-instruct-graddiff-checkpoint-8", # NOT WORKING
        "LLM-GAT/llama-3-8b-instruct-tar-checkpoint-8",
        "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint-8",
        "LLM-GAT/llama-3-8b-instruct-rmu-lat-checkpoint-8",
        "LLM-GAT/llama-3-8b-instruct-rmu-checkpoint-8",
        "LLM-GAT/llama-3-8b-instruct-rr-checkpoint-8",
        "LLM-GAT/llama-3-8b-instruct-repnoise-checkpoint-8"
    ],
    
    # TOFU models (LLaMA-2-7B)
    "tofu_models": [
        "OPTML-Group/SimNPO-TOFU-forget05-Llama-2-7b-chat",
        "OPTML-Group/SimNPO-TOFU-forget10-Llama-2-7b-chat"
    ],
    
    # MUSE models (LLaMA-2-7B)
    "muse_models": [
        "OPTML-Group/SimNPO-MUSE-News-llama-2-7b",
        "OPTML-Group/SimNPO-MUSE-Books-llama-2-7b"
    ],
    
    # Additional WMDP model
    "wmdp_zephyr": [
        "OPTML-Group/SimNPO-WMDP-zephyr-7b-beta"
    ]
}

# Define unlearning method mapping
UNLEARNING_METHODS = {
    "elm": "ELM",
    "graddiff": "GradDiff", 
    "tar": "TAR",
    "pbj": "PBJ",
    "rmu-lat": "RMU-LAT",
    "rmu": "RMU",
    "rr": "RR", 
    "repnoise": "RepNoise",
    "simnpo": "SimNPO"
}

# Model families for results aggregation
MODEL_FAMILIES = {
    "LLaMA-3-8B": MODEL_CONFIGS["wmdp_models"],
    "LLaMA-2-7B": MODEL_CONFIGS["tofu_models"] + MODEL_CONFIGS["muse_models"],
    "Zephyr-7B": MODEL_CONFIGS["wmdp_zephyr"]
}

print("Model configurations defined:")
for family, models in MODEL_FAMILIES.items():
    print(f"  {family}: {len(models)} models")
print(f"Total models to evaluate: {sum(len(models) for models in MODEL_FAMILIES.values())}")


def load_tofu_data(subset_size=50):
    """Load TOFU forget, retain, and holdout datasets"""
    print("Loading TOFU data...")

    def tofu_hf_to_dict(split):
        # Each item has 'question' and 'answer'
        return load_dataset("locuslab/TOFU", split, split='train')

    forget_prc = 10
    forget_data = tofu_hf_to_dict(f'forget{forget_prc}')
    retain_data = tofu_hf_to_dict(f'retain{100-forget_prc}')
    holdout_data = tofu_hf_to_dict('holdout10')
    
    # Limit dataset sizes
    forget_data = forget_data.select(range(min(subset_size, len(forget_data))))
    retain_data = retain_data.select(range(min(subset_size, len(retain_data))))
    holdout_data = holdout_data.select(range(min(subset_size, len(holdout_data))))
    
    print(f"TOFU data loaded: {len(forget_data)} forget, {len(retain_data)} retain, {len(holdout_data)} holdout")
    return forget_data, retain_data, holdout_data

def load_wmdp_data(subset_size=50):
    """Load WMDP forget, retain, and holdout datasets"""
    print("Loading WMDP data...")

    # Load WMDP datasets
    forget_data = load_dataset("cais/wmdp-bio-forget-corpus", split='train')
    retain_data = load_dataset("cais/wmdp-corpora", "bio-retain-corpus")['train']
    raw_holdout_data = load_dataset("cais/wmdp", "wmdp-bio", split='test')
    
    # Convert holdout format
    def convert_holdout(example):
        return {'question': example['question'], 'answer': example['choices'][example['answer']]}
    
    holdout_data = raw_holdout_data.map(convert_holdout, remove_columns=['choices'])
    
    # Limit dataset sizes
    forget_data = forget_data.select(range(min(subset_size, len(forget_data))))
    retain_data = retain_data.select(range(min(subset_size, len(retain_data))))
    holdout_data = holdout_data.select(range(min(subset_size, len(holdout_data))))
    
    print(f"WMDP data loaded: {len(forget_data)} forget, {len(retain_data)} retain, {len(holdout_data)} holdout")
    return forget_data, retain_data, holdout_data

def load_muse_data(subset_size=50, muse_type='news'):
    """Load MUSE forget, retain, and holdout datasets"""  
    print(f"Loading MUSE data ({muse_type})...")
    
    # Load MUSE datasets
    raw_or_privleak = 'raw'

    forget_file = os.path.join(MUSE_DIR, f'data/{muse_type}/{raw_or_privleak}/forget.json')
    if raw_or_privleak == 'raw':
        retain_file = os.path.join(MUSE_DIR, f'data/{muse_type}/{raw_or_privleak}/retain1.json')
    else:
        retain_file = os.path.join(MUSE_DIR, f'data/{muse_type}/{raw_or_privleak}/retain.json')
    holdout_file = os.path.join(MUSE_DIR, f'data/{muse_type}/{raw_or_privleak}/holdout.json')

    forget_data = eval_with_ILL.read_json(forget_file)
    retain_data = eval_with_ILL.read_json(retain_file)
    holdout_data = eval_with_ILL.read_json(holdout_file)
    
    # Convert to HuggingFace Dataset
    forget_data = Dataset.from_dict({'text': forget_data})
    retain_data = Dataset.from_dict({'text': retain_data})
    holdout_data = Dataset.from_dict({'text': holdout_data})

    # Limit dataset sizes
    forget_data = forget_data.select(range(min(subset_size, len(forget_data))))
    retain_data = retain_data.select(range(min(subset_size, len(retain_data))))
    holdout_data = holdout_data.select(range(min(subset_size, len(holdout_data))))
    
    print(f"MUSE data loaded: {len(forget_data)} forget, {len(retain_data)} retain, {len(holdout_data)} holdout")
    return forget_data, retain_data, holdout_data

# Define data loading functions for each benchmark
DATA_LOADERS = {
    "TOFU": load_tofu_data,
    "WMDP": load_wmdp_data,
    "MUSE_NEWS": lambda subset_size: load_muse_data(subset_size, muse_type='news'),
    "MUSE_BOOKS": lambda subset_size: load_muse_data(subset_size, muse_type='books')
}

print("Data loading functions defined for all benchmarks")


# Utility functions for dataset processing
def identify_prompt_column_type(example):
    if 'text' in example:
        return 'text'
    elif 'question' in example and 'answer' in example:
        return ['question', 'answer']
    else:
        raise ValueError("Dataset must contain either 'text' or both 'question' and 'answer' fields.")

def extract_texts_from_dataset(dataset):
    """
    Extract texts from a dataset using automatic column type detection.
    
    Args:
        dataset: List or Dataset of examples (e.g., forget_data, retain_data).
    
    Returns:
        list: List of extracted text strings.
    """
    prompt_column = identify_prompt_column_type(dataset[0])
    if isinstance(prompt_column, str):
        texts = dataset[prompt_column]
    else:
        texts = dataset[prompt_column[0]]  # Just use question for simplicity
        
    return texts

def filter_examples_by_num_tokens(example, prompt_column, tokenizer, num_min_tokens=5, max_prompt_tokens=MAX_PROMPT_TOKENS, random_num_tokens=None):
    example['valid_example'] = False  # default to invalid

    if isinstance(prompt_column, str):
        text = example.get(prompt_column, '')
        tokens = tokenizer.tokenize(text)

        if len(tokens) < num_min_tokens:
            return example

        if len(tokens) > max_prompt_tokens:
            if random_num_tokens is None:
                random_num_tokens = np.random.randint(num_min_tokens, max_prompt_tokens)
            random_num_tokens = min(random_num_tokens, len(tokens))
            text = tokenizer.convert_tokens_to_string(tokens[:random_num_tokens])
            example[prompt_column] = text

        example['valid_example'] = True
        return example

    elif isinstance(prompt_column, list) and len(prompt_column) == 2:
        question = example.get(prompt_column[0], '')
        answer = example.get(prompt_column[1], '')
        text = question # + " " + answer

        # text_tokens = tokenizer.tokenize(text)
        question_tokens = tokenizer.tokenize(question)
        answer_tokens = tokenizer.tokenize(answer)

        if len(question_tokens) < num_min_tokens:
            return example

        if len(question_tokens) > max_prompt_tokens:
            example[prompt_column[0]] = tokenizer.convert_tokens_to_string(question_tokens[:max_prompt_tokens])

        # Question is not modified
        example['valid_example'] = True
        return example

    # Unsupported format
    return example

def get_text(ex):
        """Enhanced version of get_text: Check for 'input' key first, then fall back to original logic."""
        try:
            prompt_type = identify_prompt_column_type(ex)
            if prompt_type == 'text':
                return ex['text']
            elif isinstance(prompt_type, list):
                q = ex.get('question', '').strip()
                if not q:
                    raise ValueError("Empty 'question' field")
                return q
                # return f"{ex.get('question', '')} {ex.get('answer', '')}".strip()
        except ValueError:
            # Fallback: try 'text', then 'question' + 'answer', then empty string
            warnings.warn("Fallback to empty 'question'+'answer' in get_text", UserWarning)
            return ex.get('text', f"{ex.get('question', '')} {ex.get('answer', '')}".strip())
        
def compute_auc_at_1_fp(y_true, y_proba):
    """
    Compute AUC at 1% False Positive rate using interpolation.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if len(fpr) > 1:
        interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(tpr[0], tpr[-1]))
        return float(interp_func(0.01))  # 0.01 for 1% FP
    else:
        return 0.5
    
def remove_nan_scores_and_minimize(d1, d2, d3):
        d1 = [x for x in d1 if not math.isnan(x)]
        d2 = [x for x in d2 if not math.isnan(x)]
        d3 = [x for x in d3 if not math.isnan(x)]
        min_len = min(len(d1), len(d2), len(d3))
        return d1[:min_len], d2[:min_len], d3[:min_len]
    
def chunked(iterable, batch_size):
    """Yield successive batch_size-sized chunks from iterable."""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def get_output_file_name(model_name, benchmark_name, split_name, strategy, rephrased=False):
    neighbor_method = strategy.get('neighbor_method', 'token_embedding_proximity')
    strategy_name = f"top_k_{strategy['peak_top_k']}_n_tokens_{strategy['n_tokens']}_k_neighbors_{strategy['max_neighbors']}_{neighbor_method}_MPT_{strategy['max_prompt_tokens']}_MGT_{strategy['max_gen_tokens']}".replace('[', '').replace(']', '').replace(',', '_').replace(' ', '')
    model_name_without_slash = model_name.replace('/', '-')
    
    output_file = os.path.join(PROJECT_DIR, 'data', f'subset_size_{SUBSET_SIZE}')
        
    if rephrased:
        output_file = os.path.join(output_file, 'rephrased')

    output_file = os.path.join(output_file, f'{model_name_without_slash}', strategy_name, f'exp_{benchmark_name.lower()}_neighbors', f'{split_name}.json')    
    
    return output_file
    
    
def replace_original_with_rephrased(model_name, benchmark_name, split_name, strategy, subset_size=None):
    import datasets
    output_file = get_output_file_name(model_name, benchmark_name, split_name, strategy, rephrased=False)
    
    print(f'{output_file=}')
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file {output_file} does not exist. Please run the neighbor generation script first.")
    
    print(f"Output file {output_file} already exists. Skipping generation.")
    with open(output_file, 'r') as f:
        generated_data = datasets.Dataset.from_list(json.load(f))
    
    print(f"Loaded {len(generated_data)} examples from {output_file}")
    
    # Filter to keep only valid examples where 'valid_example' is True
    filtered_dataset = generated_data.filter(lambda example: example['valid_example'])
    
    if subset_size is not None:
        filtered_dataset = filtered_dataset.select(range(min(subset_size, len(filtered_dataset))))
        
    column_to_rephrase = identify_prompt_column_type(generated_data[0])
    
    # Create a new dataset with only the column_to_rephrase, replacing original with 1st neighbor
    def replace_with_first_neighbor(example):
        if 'neighbors' in example and len(example['neighbors']) > 1:  # Need at least 2 neighbors to start from second
            for i in range(1, min(6, len(example['neighbors']))):  # Start from index 1 (second neighbor), up to 5 total checks
                neighbor = example['neighbors'][i]
                is_identical = False
                if isinstance(column_to_rephrase, str):
                    # If column_to_rephrase is 'text'
                    if example[column_to_rephrase] == neighbor.get(column_to_rephrase, example[column_to_rephrase]):
                        is_identical = True
                        print(f"Identical neighbor found at neighbor index {i}")
                elif isinstance(column_to_rephrase, list):
                    # If column_to_rephrase is ['question', 'answer']
                    # Since answers are identical, check if question (neighbor['text']) matches original question
                    if neighbor.get('text', example[column_to_rephrase[0]]) == example[column_to_rephrase[0]]:
                        is_identical = True
                        print(f"Identical neighbor found at neighbor index {i}")
                if not is_identical:
                    # Use this neighbor
                    if isinstance(column_to_rephrase, str):
                        example[column_to_rephrase] = neighbor.get(column_to_rephrase, example[column_to_rephrase])
                    elif isinstance(column_to_rephrase, list):
                        # Replace question with neighbor's text, keep answer the same
                        example[column_to_rephrase[0]] = neighbor.get('text', example[column_to_rephrase[0]])
                        # example[column_to_rephrase[1]] = example[column_to_rephrase[1]]
                        # example[column_to_rephrase[1]] remains unchanged as answers are identical
                    break  # Found a non-identical one, stop
            # If all checked neighbors are identical, keep the original (no change)
        return example
    
    # Apply the replacement
    rephrased_dataset = filtered_dataset.map(replace_with_first_neighbor)
    
    # Select only the column_to_rephrase columns
    if isinstance(column_to_rephrase, str):
        selected_columns = [column_to_rephrase]
    else:
        selected_columns = column_to_rephrase
    
    # Add other necessary columns if needed, but keep only the rephrased text columns
    final_dataset = rephrased_dataset.select_columns(selected_columns)
    
    return final_dataset
    
    
    
def load_neighbors_data(model_name, benchmark_name, split_name, strategy):
    import datasets

    output_file = get_output_file_name(model_name, benchmark_name, split_name, strategy, rephrased=REPHRASED_ORIGINAL)

    
    print(f'{output_file=}')
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file {output_file} does not exist. Please run the neighbor generation script first.")
    
    print(f"Output file {output_file} already exists. Skipping generation.")
    with open(output_file, 'r') as f:
        generated_data = datasets.Dataset.from_list(json.load(f))
    
    print(f"Loaded {len(generated_data)} examples from {output_file}")
    
    # Filter to keep only valid examples where 'valid_example' is True
    filtered_dataset = generated_data.filter(lambda example: example['valid_example'])

    # column = identify_prompt_column_type(filtered_dataset[0])
    # if isinstance(column, str):
    #     column = [column]
    # # Select only the desired columns: 'text', 'neighbors', 'mean_neighbors_loss'
    # selected_dataset = filtered_dataset.select_columns(column+['neighbors', 'mean_neighbors_loss'])

    return filtered_dataset

def compute_all_aucs(forget_scores, retain_scores, holdout_scores):
    # Convert inputs to lists if they are tensors
    def to_list(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().tolist()
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, list):
            return x

    forget_scores = to_list(forget_scores)
    retain_scores = to_list(retain_scores)
    holdout_scores = to_list(holdout_scores)

    # Combine scores and labels
    all_scores = forget_scores + retain_scores + holdout_scores
    all_labels = [1] * len(forget_scores) + [0] * (len(retain_scores) + len(holdout_scores))
    
    # # Multi-class labels for per-class ROC AUC
    # multiclass_labels = (
    #     [2] * len(forget_scores) + 
    #     [1] * len(retain_scores) + 
    #     [0] * len(holdout_scores)
    # )

    retain_vs_all_labels = [0]*len(forget_scores) + [1]*len(retain_scores) + [0]*len(holdout_scores)
    forget_vs_all_labels = [1]*len(forget_scores) + [0]*len(retain_scores) + [0]*len(holdout_scores)
    holdout_vs_all_labels = [0]*len(forget_scores) + [0]*len(retain_scores) + [1]*len(holdout_scores)

    # Compute binary AUCs
    retain_vs_all_auc = roc_auc_score(retain_vs_all_labels, all_scores)
    forget_vs_all_auc = roc_auc_score(forget_vs_all_labels, all_scores)
    holdout_vs_all_auc = roc_auc_score(holdout_vs_all_labels, all_scores)
    
    # Compute multi-class AUC as average of binary AUCs (fix for 1D scores)
    multi_class_auc = (retain_vs_all_auc + forget_vs_all_auc + holdout_vs_all_auc) / 3

    # Compute AUC at 1% FP for each binary comparison
    def compute_auc_at_1_fp_local(y_true, y_proba):
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        if len(fpr) > 1:
            interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(tpr[0], tpr[-1]))
            return float(interp_func(0.01))  # 0.01 for 1% FP
        else:
            return 0.5

    retain_vs_all_auc_at_1_fp = compute_auc_at_1_fp_local(retain_vs_all_labels, all_scores)
    forget_vs_all_auc_at_1_fp = compute_auc_at_1_fp_local(forget_vs_all_labels, all_scores)
    holdout_vs_all_auc_at_1_fp = compute_auc_at_1_fp_local(holdout_vs_all_labels, all_scores)

    return {
        'retain_vs_all_auc': retain_vs_all_auc,
        'forget_vs_all_auc': forget_vs_all_auc,
        'holdout_vs_all_auc': holdout_vs_all_auc,
        'multi_class_auc': multi_class_auc,
        'retain_vs_all_auc_at_1_fp': retain_vs_all_auc_at_1_fp,
        'forget_vs_all_auc_at_1_fp': forget_vs_all_auc_at_1_fp,
        'holdout_vs_all_auc_at_1_fp': holdout_vs_all_auc_at_1_fp
    }

def compute_spv_mia_auc(forget_data: torch.tensor, retain_data: torch.tensor, holdout_data: torch.tensor, features_names: list):
    """Compute the SPV-MIA AUC score.

    NOTE: in the data there are already the original, neighbors, and each one loss.

    Args:
        forget_data (torch.tensor): [num_forget, num_features]
        retain_data (torch.tensor): [num_retain, num_features]
        holdout_data (torch.tensor): [num_holdout, num_features]
        model (torch.nn.Module): The model to evaluate.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text encoding.
        batch_size (int, optional): The batch size for evaluation. Defaults to 64.
    """
    neigh_stats_methods = ['max', 'mean']

    max_features_idx = features_names.index('max_neighbor_loss')
    mean_features_idx = features_names.index('mean_neighbor_loss')
    orig_loss_idx = features_names.index('original_loss')
    
    def compute_scores(data, idx):
        orig_losses = data[:, orig_loss_idx]
        neigh_losses = data[:, idx]
        scores = orig_losses - neigh_losses
        return scores.tolist()
    
    results = {}
    for stat_method in neigh_stats_methods:
        if stat_method == 'max':
            forget_scores = compute_scores(forget_data, max_features_idx)
            retain_scores = compute_scores(retain_data, max_features_idx)
            holdout_scores = compute_scores(holdout_data, max_features_idx)
        elif stat_method == 'mean':
            forget_scores = compute_scores(forget_data, mean_features_idx)
            retain_scores = compute_scores(retain_data, mean_features_idx)
            holdout_scores = compute_scores(holdout_data, mean_features_idx)
            
        results[stat_method] = compute_all_aucs(forget_scores, retain_scores, holdout_scores)
            
            

    # Compute AUCs for each statistic
    return results

def compute_loss_based_auc(forget_data: torch.tensor, retain_data: torch.tensor, holdout_data: torch.tensor, features_names: list):
    orig_loss_idx = features_names.index('original_loss')
    forget_scores = forget_data[:, orig_loss_idx]
    retain_scores = retain_data[:, orig_loss_idx]
    holdout_scores = holdout_data[:, orig_loss_idx]
    
    return compute_all_aucs(forget_scores, retain_scores, holdout_scores)

def compute_zlib_compression_auc(forget_data, retain_data, holdout_data):
    forget_orig_losses = Dataset.from_list(forget_data['original_output'])
    retain_orig_losses = Dataset.from_list(retain_data['original_output'])
    holdout_orig_losses = Dataset.from_list(holdout_data['original_output'])

    forget_compressed_lengths = forget_data.map(lambda x: {'compressed_length': len(zlib.compress(get_text(x).encode('utf-8')))})
    retain_compressed_lengths = retain_data.map(lambda x: {'compressed_length': len(zlib.compress(get_text(x).encode('utf-8')))} )
    holdout_compressed_lengths = holdout_data.map(lambda x: {'compressed_length': len(zlib.compress(get_text(x).encode('utf-8')))} )

    forget_scores = [ -x['loss'] / y['compressed_length'] for x, y in zip(forget_orig_losses, forget_compressed_lengths)]
    retain_scores = [ -x['loss'] / y['compressed_length'] for x, y in zip(retain_orig_losses, retain_compressed_lengths)]
    holdout_scores = [ -x['loss'] / y['compressed_length'] for x, y in zip(holdout_orig_losses, holdout_compressed_lengths)]
        
    return compute_all_aucs(forget_scores, retain_scores, holdout_scores)


# def compute_zlib_compression_auc(forget_data, retain_data, holdout_data, model, tokenizer, batch_size=128):
#     def compute_scores(data):
#         # Extract all texts once before the loop for efficiency
#         texts = [get_text(d) for d in data]
        
#         scores = []
        
#         for text in tqdm(texts, desc="Zlib Compression"):
#             inputs = tokenizer([text], return_tensors='pt', padding=True, truncation=True)
#             input_ids = inputs['input_ids'].to(model.device)
#             attention_mask = inputs['attention_mask'].to(model.device)

#             with torch.no_grad():
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
#                 loss = outputs.loss.item()  # Single loss for this input

#             ll = -loss
#             compressed_len = len(zlib.compress(text.encode('utf-8')))
#             scores.append(ll / compressed_len)
#         return scores
    # 
    # 
    # forget_scores = compute_scores(forget_data)
    # retain_scores = compute_scores(retain_data)
    # holdout_scores = compute_scores(holdout_data)

    # return compute_all_aucs(forget_scores, retain_scores, holdout_scores)

def compute_min_k_auc(forget_data, retain_data, holdout_data, model, tokenizer, k_percent=0.1, batch_size=64):
    def compute_scores(data):
        texts = [get_text(d) for d in data]
        scores = []

        def run_batch(batch):
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = enc['input_ids'].to(model.device)
            attention_mask = enc['attention_mask'].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits  # [B, T, V]
            return input_ids, logits

        def compute_min_k_for_sequence(input_ids, logits, k_percent=0.1):
            log_probs = F.log_softmax(logits, dim=-1)  # [T, V]
            log_probs = log_probs[:-1]       # [T-1, V]
            target_ids = input_ids[1:len(log_probs)+1].unsqueeze(-1)  # [T-1, 1]
            token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [T-1]
            k_len = max(1, int(len(token_log_probs) * k_percent))
            min_k_mean = torch.topk(token_log_probs, k=k_len, largest=False).values.mean().item()
            return min_k_mean

        for batch in tqdm(chunked(texts, batch_size), desc=f"min-k ({k_percent*100:.0f}%)"):
            input_ids, logits = run_batch(batch)
            max_attempts = 10
            for i in tqdm(range(len(batch)), desc="min-k attempts", leave=False, total=len(batch)):
                attempt = 0
                min_k_mean = compute_min_k_for_sequence(input_ids[i], logits[i], k_percent)
                while not math.isfinite(min_k_mean) and attempt < max_attempts:
                    single_input_ids, single_logits = run_batch([batch[i]])
                    min_k_mean = compute_min_k_for_sequence(single_input_ids[0], single_logits[0], k_percent)
                    attempt += 1
                if not math.isfinite(min_k_mean):
                    min_k_mean = float('nan')
                scores.append(min_k_mean)
        return scores

    forget_scores = compute_scores(forget_data)
    retain_scores = compute_scores(retain_data)
    holdout_scores = compute_scores(holdout_data)
    forget_scores, retain_scores, holdout_scores = remove_nan_scores_and_minimize(forget_scores, retain_scores, holdout_scores)

    return compute_all_aucs(forget_scores, retain_scores, holdout_scores)

def compute_min_k_pp_auc(forget_data, retain_data, holdout_data, model, tokenizer, k_percent=0.1, batch_size=64):            
    def compute_scores(data):
        # Extract all texts once before the loop for efficiency
        texts = [get_text(d) for d in data]
        
        scores = []
        
        def run_batch(batch):
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = enc['input_ids'].to(model.device)
            attention_mask = enc['attention_mask'].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                logits = outputs.logits  # [B, T, V]
            
            return input_ids, logits
        
        def compute_min_kpp_for_sequence(input_ids, logits, k_percent=0.1):
            """
            Compute the MIN-K%++ score for a single sequence.

            Args:
                input_ids (torch.Tensor): Shape [T], token ids for the sequence.
                logits (torch.Tensor): Shape [T, V], logits for each token position.
                k_percent (float): Fraction of lowest-scoring tokens to average.

            Returns:
                float: The mean of the bottom k% normalized log-prob scores (MIN-K%++).
            """
            # Compute log-probs and probs
            log_probs = F.log_softmax(logits, dim=-1)  # [T, V]
            probs = F.softmax(logits, dim=-1)          # [T, V]

            # Shift for next-token prediction
            log_probs = log_probs[:-1]                 # [T-1, V]
            probs = probs[:-1]                         # [T-1, V]
            target_ids = input_ids[1:len(log_probs)+1].unsqueeze(-1)  # [T-1, 1]
            token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [T-1]

            # Compute mean and std for normalization
            mu = (probs * log_probs).sum(-1)  # [T-1]
            sigma = (probs * (log_probs ** 2)).sum(-1) - mu ** 2
            sigma = torch.sqrt(sigma + 1e-8)  # [T-1]

            # Normalized log-prob (MIN-K%++)
            mink_pp = (token_log_probs - mu) / sigma  # [T-1]

            # Take mean of bottom k%
            k_len = max(1, int(len(mink_pp) * k_percent))
            min_kpp_mean = torch.topk(mink_pp, k=k_len, largest=False).values.mean().item()
            return min_kpp_mean

        for batch in tqdm(chunked(texts, batch_size), desc=f"min-k++ ({k_percent*100:.0f}%)"):
            input_ids, logits = run_batch(batch)

            max_attempts = 10
        
            for i in range(len(batch)):
                attempt = 0
                min_kpp_mean = compute_min_kpp_for_sequence(input_ids[i], logits[i], k_percent)
                
                while not math.isfinite(min_kpp_mean) and attempt < max_attempts:
                    # run this sequence again individually
                    single_input_ids, single_logits = run_batch([batch[i]])
                    min_kpp_mean = compute_min_kpp_for_sequence(single_input_ids[0], single_logits[0], k_percent)
                    attempt += 1
                    
                if not math.isfinite(min_kpp_mean):
                    min_kpp_mean = float('nan')

                scores.append(min_kpp_mean)
                
        return scores
    
    forget_scores = compute_scores(forget_data)
    retain_scores = compute_scores(retain_data)
    holdout_scores = compute_scores(holdout_data)
    
    forget_scores, retain_scores, holdout_scores = remove_nan_scores_and_minimize(forget_scores, retain_scores, holdout_scores)

    return compute_all_aucs(forget_scores, retain_scores, holdout_scores)



def compute_rouge_l_f1_auc(forget_data, retain_data, holdout_data, model, tokenizer, batch_size=128):
    """
    Compute ROUGE-L F1 based AUC using generated vs. reference answers.
    """
    def get_rouge_scores_batch(qa_pairs, model, tokenizer, batch_size=128, desc="Processing"):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        all_scores = []
        
        # Ensure left padding for decoder-only models
        if model.config.is_decoder and not getattr(model.config, 'is_encoder_decoder', False):
            tokenizer.padding_side = 'left'
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(qa_pairs), batch_size), desc=desc):
            batch_qa = qa_pairs[i:i+batch_size]
            batch_questions = []
            batch_ref_answers = []
            batch_input_ids = []
            batch_attention_masks = []
            
            # Prepare batch
            for question, ref_answer in batch_qa:
                inputs = tokenizer(
                    question, 
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                ).to(model.device)
                input_ids = inputs['input_ids'][0]
                
                if not ref_answer.strip():
                    # Take 0.3 of the question as fallback
                    len_ans_ids = int(0.3 * len(input_ids))
                    ref_answer_ids = input_ids[-len_ans_ids:]
                    ref_answer = tokenizer.decode(ref_answer_ids, skip_special_tokens=True).strip()
                    # Remove fallback answer from prompt for generation
                    input_ids = input_ids[:-len_ans_ids]
                    attention_mask = inputs['attention_mask'][0][:-len_ans_ids]
                else:
                    attention_mask = inputs['attention_mask'][0]
                
                batch_questions.append(question)
                batch_ref_answers.append(ref_answer)
                batch_input_ids.append(input_ids)
                batch_attention_masks.append(attention_mask)
            
            # Pad the batch to the same length for batched generation
            max_len = max(len(ids) for ids in batch_input_ids)
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(batch_input_ids, batch_attention_masks):
                pad_len = max_len - len(ids)
                if pad_len > 0:
                    # Pad with pad_token_id (left padding for decoder-only models)
                    padded_ids = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device), ids])
                    padded_mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype, device=mask.device), mask])
                else:
                    padded_ids = ids
                    padded_mask = mask
                padded_input_ids.append(padded_ids)
                padded_attention_masks.append(padded_mask)
            
            # Stack into tensors for batched generation
            batched_input_ids = torch.stack(padded_input_ids)
            batched_attention_masks = torch.stack(padded_attention_masks)
            
            # First loop: Generate answers for the entire batch at once
            batch_gen_answers = []
            max_attempts = 10
            attempt = 0
            while len(batch_gen_answers) < len(batch_qa) and attempt < max_attempts:
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=batched_input_ids,
                            attention_mask=batched_attention_masks,
                            max_new_tokens=MAX_GEN_TOKENS, 
                            do_sample=False, 
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        # Decode answers for all examples in the batch
                        for j in range(len(batch_qa)):
                            input_length = len(batch_input_ids[j])
                            generated_tokens = outputs[j][input_length:]
                            gen_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                            batch_gen_answers.append(gen_answer)
                        break  # Success, exit attempt loop
                except Exception as e:
                    print(f"Generation error: {e}")
                    attempt += 1
                    if attempt >= max_attempts:
                        # Fill remaining with empty strings on failure
                        while len(batch_gen_answers) < len(batch_qa):
                            batch_gen_answers.append("")
            
            # Second loop: Handle retries for empty answers individually
            for j in range(len(batch_qa)):
                question, ref_answer = batch_questions[j], batch_ref_answers[j]
                gen_answer = batch_gen_answers[j]
                if not gen_answer:  # If empty, retry individually
                    attempt = 0
                    while not gen_answer and attempt < max_attempts:
                        try:
                            inputs = tokenizer(
                                question, 
                                return_tensors='pt',
                                max_length=512,
                                truncation=True
                            ).to(model.device)
                            with torch.no_grad():
                                outputs = model.generate(
                                    input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    max_new_tokens=MAX_GEN_TOKENS, 
                                    do_sample=False, 
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id
                                )
                                # Decode only the generated part
                                input_length = inputs['input_ids'].shape[1]
                                generated_tokens = outputs[0][input_length:]
                                gen_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        except Exception as e:
                            print(f"Retry generation error: {e}")
                            pass
                        attempt += 1
                    batch_gen_answers[j] = gen_answer  # Update with retried answer
            
            # Now compute ROUGE-L F1 scores for all examples
            batch_scores = []
            for j, (question, ref_answer, gen_answer) in enumerate(zip(batch_questions, batch_ref_answers, batch_gen_answers)):
                if gen_answer:
                    score = scorer.score(ref_answer, gen_answer)['rougeL'].fmeasure
                    batch_scores.append(score)
                else:
                    batch_scores.append(None)
            
            all_scores.extend(batch_scores)
        
        return all_scores
    
    # Helper to extract question-answer pairs using modular functions
    def extract_qa_pairs(dataset):
        if not dataset:
            return []
        try:
            prompt_column = identify_prompt_column_type(dataset[0])
            qa_pairs = []
            for ex in dataset:
                if isinstance(prompt_column, list) and len(prompt_column) == 2:  # 'question' and 'answer'
                    question = ex.get(prompt_column[0], "")
                    answer = ex.get(prompt_column[1], "")
                elif prompt_column == 'text':  # 'text' case: treat as question, empty answer
                    question = ex.get('text', "")
                    answer = ""  # No reference answer for text-based datasets
                else:
                    question = ""
                    answer = ""
                qa_pairs.append((question, answer))
            return qa_pairs
        except ValueError as e:
            print(f"Warning: {e}. Returning empty QA pairs.")
            return []
    
    # Extract QA pairs for each dataset
    forget_qa = extract_qa_pairs(forget_data)
    retain_qa = extract_qa_pairs(retain_data)
    holdout_qa = extract_qa_pairs(holdout_data)
    
    print("Computing ROUGE-L F1 scores...")
    # Compute scores in batches with progress bars
    forget_scores = get_rouge_scores_batch(forget_qa, model, tokenizer, batch_size, desc="Forget ROUGE-L")
    retain_scores = get_rouge_scores_batch(retain_qa, model, tokenizer, batch_size, desc="Retain ROUGE-L")
    holdout_scores = get_rouge_scores_batch(holdout_qa, model, tokenizer, batch_size, desc="Holdout ROUGE-L")
    
    # Remove None scores
    forget_scores = [s for s in forget_scores if s is not None]
    retain_scores = [s for s in retain_scores if s is not None]
    holdout_scores = [s for s in holdout_scores if s is not None]
    
    # Subset to the same size
    min_size = min(len(forget_scores), len(retain_scores), len(holdout_scores))
    forget_scores = forget_scores[:min_size]
    retain_scores = retain_scores[:min_size]
    holdout_scores = holdout_scores[:min_size]
    
    # Train classifiers
    print("Training classifiers...")
    # Retain vs All (retain=1, others=0)
    X_ra = np.array(retain_scores + forget_scores + holdout_scores).reshape(-1, 1)
    y_ra = np.array([1] * len(retain_scores) + [0] * (len(forget_scores) + len(holdout_scores)))
    clf_ra = LogisticRegression().fit(X_ra, y_ra)
    retain_vs_all_auc = roc_auc_score(y_ra, clf_ra.predict_proba(X_ra)[:, 1])
    retain_vs_all_auc_at_1_fp = compute_auc_at_1_fp(y_ra, clf_ra.predict_proba(X_ra)[:, 1])

    # Forget vs All (forget=1, others=0)
    X_fa = np.array(forget_scores + retain_scores + holdout_scores).reshape(-1, 1)
    y_fa = np.array([1] * len(forget_scores) + [0] * (len(retain_scores) + len(holdout_scores)))
    clf_fa = LogisticRegression().fit(X_fa, y_fa)
    forget_vs_all_auc = roc_auc_score(y_fa, clf_fa.predict_proba(X_fa)[:, 1])
    forget_vs_all_auc_at_1_fp = compute_auc_at_1_fp(y_fa, clf_fa.predict_proba(X_fa)[:, 1])

    # Holdout vs All (holdout=1, others=0)
    X_ha = np.array(holdout_scores + retain_scores + forget_scores).reshape(-1, 1)
    y_ha = np.array([1] * len(holdout_scores) + [0] * (len(retain_scores) + len(forget_scores)))
    clf_ha = LogisticRegression().fit(X_ha, y_ha)
    holdout_vs_all_auc = roc_auc_score(y_ha, clf_ha.predict_proba(X_ha)[:, 1])
    holdout_vs_all_auc_at_1_fp = compute_auc_at_1_fp(y_ha, clf_ha.predict_proba(X_ha)[:, 1])

    # Multiclass AUC: 3-class classifier (forget=0, retain=1, holdout=2)
    X_multi = np.array(forget_scores + retain_scores + holdout_scores).reshape(-1, 1)
    y_multi = np.array([0] * len(forget_scores) + [1] * len(retain_scores) + [2] * len(holdout_scores))
    clf_multi = LogisticRegression(multi_class='ovr').fit(X_multi, y_multi)
    multi_class_auc = roc_auc_score(y_multi, clf_multi.predict_proba(X_multi), multi_class='ovr')

    return {
        'retain_vs_all_auc': retain_vs_all_auc,
        'forget_vs_all_auc': forget_vs_all_auc,
        'holdout_vs_all_auc': holdout_vs_all_auc,
        'multi_class_auc': multi_class_auc,
        'retain_vs_all_auc_at_1_fp': retain_vs_all_auc_at_1_fp,
        'forget_vs_all_auc_at_1_fp': forget_vs_all_auc_at_1_fp,
        'holdout_vs_all_auc_at_1_fp': holdout_vs_all_auc_at_1_fp
    }

def run_ill_evaluation(model_name, benchmark_name, model, tokenizer, datasets, neighbor_method, test_size=TEST_SIZE):
    """
    Run Input Loss Landscape (ILL) evaluation for a given model and benchmark.
    
    Args:
        model_name: HuggingFace model name/path
        benchmark_name: Benchmark name (TOFU, WMDP, MUSE)
        model: Loaded model
        tokenizer: Associated tokenizer
        datasets: Dictionary of datasets (forget, retain, holdout)
        neighbor_method: Neighbor generation method
        model_configs: Model configuration dictionary
        cosine_similarities_file: Path to cosine similarities file
        
    Returns:
        tuple: (results, binary_results, feature_importance_results, binary_feature_importance)
    """
    # Update strategy with the neighbor method
    strategy = STRATEGY.copy()
    strategy['method'] = neighbor_method
    print(f"{strategy=}")
    
    if benchmark_name == "TOFU":
        model_configs = {
            'question_start_tag': '[INST]',
            'question_end_tag': ' [/INST]',
            'answer_tag': ''
        }
    elif benchmark_name == "WMDP":
        model_configs = {
            'question_start_tag': 'Question: ',
            'question_end_tag': '\n',
            'answer_tag': 'Answer: '
        }
    else:  # default MUSE
        model_configs = {
            'question_start_tag': 'Question: ',
            'question_end_tag': '\n',
            'answer_tag': 'Answer: '
        }

    output_dirs = {'neighbors': f'exp_{benchmark_name.lower()}_neighbors'}
    if REPHRASED_ORIGINAL:
        output_dirs['rephrased_original'] = 'rephrased'
    # Set up ILL evaluation kwargs
    new_ILL_eval_kwargs = {
        'model_name': model_name,
        'model': model,
        'tokenizer': tokenizer,
        'datasets': datasets,
        'prompt_column': ['question', 'answer'],
        'create_new_neighbors_file': True,
        'showplts': False,
        'plots_output_dir': None,
        'strategy': strategy,
        'output_dirs': output_dirs,
        'model_configs': model_configs,
        'max_prompt_tokens': MAX_PROMPT_TOKENS,
        'max_gen_tokens': MAX_GEN_TOKENS,
        'output_file': get_output_file_name(model_name, benchmark_name, 'SPLITS_HOLDER', strategy, rephrased=REPHRASED_ORIGINAL)
    }
    
    print("Running ILL evaluation...")
    features_dict = new_ILL_eval(new_ILL_eval_kwargs)
    
    # Extract feature tensors
    forget_tensor = features_dict['forget']['unnormalized_features_tensor']
    retain_tensor = features_dict['retain']['unnormalized_features_tensor']
    holdout_tensor = features_dict['holdout']['unnormalized_features_tensor']
    
    # Ensure consistent sizes
    min_examples = min(len(forget_tensor), len(retain_tensor), len(holdout_tensor))
    forget_tensor = forget_tensor[:min_examples]
    retain_tensor = retain_tensor[:min_examples] 
    holdout_tensor = holdout_tensor[:min_examples]
    
    print(f"Feature tensors: forget {forget_tensor.shape}, retain {retain_tensor.shape}, holdout {holdout_tensor.shape}")
    
    # Normalize features
    norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor = eval_with_ILL.normalize_features(
        forget_tensor, retain_tensor, holdout_tensor
    )
    
    # Get feature labels
    features_labels = features_dict['forget']['features_names']
    
    trained_classifiers = {}
    
    # Train classifiers
    print("Training predictors...")
    results, feature_importance_results, classifiers = eval_with_ILL.train_predictors(
        retain_t=norm_retain_tensor,
        holdout_t=norm_holdout_tensor,
        features_labels=features_labels,
        forget_t=norm_forget_tensor,
        test_size=test_size
    )

    trained_classifiers['overall_predictors'] = classifiers
    
    # Train binary comparisons
    binary_results, binary_feature_importance, binary_classifiers = eval_with_ILL.train_binary_comparisons(norm_retain_tensor, norm_holdout_tensor, norm_forget_tensor, features_labels, test_size=test_size)
    
    trained_classifiers['binary_comparisons'] = binary_classifiers
    
    # Add custom binary comparisons for table generation
    def train_custom_binary_comparison(tensor1, tensor2, labels):
        """Train binary classifier between two specific tensors"""
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        X = torch.cat([tensor1, tensor2]).numpy()
        y = torch.cat([torch.ones(len(tensor1)), torch.zeros(len(tensor2))]).numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        results = {}
        classifiers = {}
        for clf_type, (clf_class, params) in [
            ('logistic', (LogisticRegression, {'random_state': 42, 'max_iter': 300})), 
            ('random_forest', (RandomForestClassifier, {'random_state': 42, 'n_estimators': 100}))
        ]:
            clf = clf_class(**params).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)

            # Interpolate to find TPR at FPR = 0.01 (1%)
            if len(fpr) > 1:
                interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=(tpr[0], tpr[-1]))
                auc_at_1_fp = float(interp_func(0.01))
            else:
                auc_at_1_fp = 0.5

            results[clf_type] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'auc_at_1_fp': auc_at_1_fp  # Add this for AUC at 1% FP
            }
            classifiers[clf_type] = clf
        return results, classifiers

    # Add the specific comparisons we need for the tables
    binary_results['retain_vs_all'], classifiers = train_custom_binary_comparison( norm_retain_tensor, torch.cat([norm_forget_tensor, norm_holdout_tensor]), features_labels)
    trained_classifiers['retain_vs_all'] = classifiers
    
    binary_results['forget_vs_all'], classifiers = train_custom_binary_comparison(norm_forget_tensor, torch.cat([norm_retain_tensor, norm_holdout_tensor]), features_labels)
    trained_classifiers['forget_vs_all'] = classifiers
    
    binary_results['holdout_vs_all'], classifiers = train_custom_binary_comparison( norm_holdout_tensor, torch.cat([norm_retain_tensor, norm_forget_tensor]), features_labels)
    trained_classifiers['holdout_vs_all'] = classifiers
    
    # Add pairwise comparisons
    binary_results['retain_vs_forget'], classifiers = train_custom_binary_comparison(norm_retain_tensor, norm_forget_tensor, features_labels)
    trained_classifiers['retain_vs_forget'] = classifiers
    
    binary_results['retain_vs_holdout'], classifiers = train_custom_binary_comparison(norm_retain_tensor, norm_holdout_tensor, features_labels)
    trained_classifiers['retain_vs_holdout'] = classifiers
    
    binary_results['forget_vs_holdout'], classifiers = train_custom_binary_comparison(norm_forget_tensor, norm_holdout_tensor, features_labels)
    trained_classifiers['forget_vs_holdout'] = classifiers

    # Compute LOSS-based AUCs
    if COMPUTE_BASELINES and neighbor_method == 'token_embedding_proximity': # we dont need to run it 4 times. the original loss is the same
        binary_results['loss_based'] = compute_loss_based_auc(norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor, features_labels)
        
    # Compute SPV-MIA AUCs
    if COMPUTE_BASELINES:
        try:
            binary_results['spv_mia'] = compute_spv_mia_auc(norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor, features_labels)
        except Exception as e:
            print(f"❌ Error computing SPV-MIA AUC: {e}")
            binary_results['spv_mia'] = {'retain_vs_all_auc': 0,
                        'forget_vs_all_auc': 0,
                        'holdout_vs_all_auc': 0,
                        'multi_class_auc': 0,
                        'retain_vs_all_auc_at_1_fp': 0,
                        'forget_vs_all_auc_at_1_fp': 0,
                        'holdout_vs_all_auc_at_1_fp': 0
                        }
    
    # Perform HDBSCAN clustering and visualization
    try:
        print("\n🔍 Performing HDBSCAN clustering and visualization...")
        
        # Perform clustering with default parameters
        clustering_results = perform_hdbscan_clustering(
            norm_forget_tensor,
            norm_retain_tensor,
            norm_holdout_tensor,
            min_cluster_size=10,
            min_samples=10
        )
        
        # Create 2D visualization
        plot_title = f"({model_name.split('/')[-1]}, {benchmark_name}, {neighbor_method})"
        fig_2d = plot_hdbscan_results(
            norm_forget_tensor,
            norm_retain_tensor,
            norm_holdout_tensor,
            clustering_results,
            plot_dim='2d',
            save_path=None,  # Can be updated to save to dm
            title_suffix=plot_title
        )
        
        # Create 3D visualization
        fig_3d = plot_hdbscan_results(
            norm_forget_tensor,
            norm_retain_tensor,
            norm_holdout_tensor,
            clustering_results,
            plot_dim='3d',
            save_path=None,  # Can be updated to save to dm
            title_suffix=plot_title
        )
        
        fig_2d_filename = f"{model_name.replace('/', '_')}_{benchmark_name}_{neighbor_method}_hdbscan_2d.png"
        fig_3d_filename = f"{model_name.replace('/', '_')}_{benchmark_name}_{neighbor_method}_hdbscan_3d.png"
        clustering_results['figs'] = {'2d': {'name': fig_2d_filename, 'fig': fig_2d},
                                      '3d': {'name': fig_3d_filename, 'fig': fig_3d}}
        
        plt.show()
    except Exception as e:
        print(f"⚠️ Warning: Could not perform HDBSCAN clustering: {e}")
        import traceback
        traceback.print_exc()

    return results, binary_results, feature_importance_results, binary_feature_importance, trained_classifiers, clustering_results, features_dict


def perform_hdbscan_clustering(forget_tensor, retain_tensor, holdout_tensor, min_cluster_size=3, min_samples=3):
    """
    Perform HDBSCAN clustering on the combined feature tensors.
    
    Args:
        forget_tensor: Feature tensor for forget set
        retain_tensor: Feature tensor for retain set  
        holdout_tensor: Feature tensor for holdout set
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples for HDBSCAN
        
    Returns:
        dict: Dictionary containing:
            - 'labels': Cluster labels for all samples
            - 'probabilities': Cluster membership probabilities
            - 'forget_labels': Cluster labels for forget samples
            - 'retain_labels': Cluster labels for retain samples
            - 'holdout_labels': Cluster labels for holdout samples
            - 'n_clusters': Number of clusters found
            - 'n_noise': Number of noise points
    """
    print("\n" + "="*60)
    print("Performing HDBSCAN Clustering")
    print("="*60)
    
    # Combine all tensors
    combined_tensor = torch.cat([forget_tensor, retain_tensor, holdout_tensor]).numpy()
    forget_len = len(forget_tensor)
    retain_len = len(retain_tensor)
    holdout_len = len(holdout_tensor)
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}...")
    cluster_labels = clusterer.fit_predict(combined_tensor)
    probabilities = clusterer.probabilities_
    
    # Split labels back to original sets
    forget_labels = cluster_labels[:forget_len]
    retain_labels = cluster_labels[forget_len:forget_len+retain_len]
    holdout_labels = cluster_labels[forget_len+retain_len:]
    
    # Count clusters and noise
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nClustering Results:")
    print(f"  Number of clusters found: {n_clusters}")
    print(f"  Number of noise points: {n_noise} ({100*n_noise/len(cluster_labels):.1f}%)")
    print(f"  Forget set: {len(set(forget_labels))} unique clusters")
    print(f"  Retain set: {len(set(retain_labels))} unique clusters")
    print(f"  Holdout set: {len(set(holdout_labels))} unique clusters")
    
    return {
        'labels': cluster_labels,
        'probabilities': probabilities,
        'forget_labels': forget_labels,
        'retain_labels': retain_labels,
        'holdout_labels': holdout_labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'clusterer': clusterer
    }


def plot_hdbscan_results(forget_tensor, retain_tensor, holdout_tensor, clustering_results, 
                         plot_dim='2d', save_path=None, title_suffix=''):
    """
    Visualize HDBSCAN clustering results in 2D or 3D.
    
    Args:
        forget_tensor: Feature tensor for forget set
        retain_tensor: Feature tensor for retain set
        holdout_tensor: Feature tensor for holdout set
        clustering_results: Results from perform_hdbscan_clustering
        plot_dim: '2d' or '3d' for visualization dimension
        save_path: Path to save the plot (optional)
        title_suffix: Additional text to add to plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print(f"\nCreating {plot_dim.upper()} visualization...")
    
    # Combine tensors and convert to numpy
    combined_tensor = torch.cat([forget_tensor, retain_tensor, holdout_tensor]).numpy()
    forget_len = len(forget_tensor)
    retain_len = len(retain_tensor)
    
    # Determine number of components for PCA
    n_components = 3 if plot_dim == '3d' else 2
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(combined_tensor)
    
    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_var}")
    print(f"Total variance explained: {sum(explained_var):.2%}")
    
    # Split back into sets
    forget_reduced = reduced_data[:forget_len]
    retain_reduced = reduced_data[forget_len:forget_len+retain_len]
    holdout_reduced = reduced_data[forget_len+retain_len:]
    
    # Get cluster labels
    cluster_labels = clustering_results['labels']
    forget_labels = clustering_results['forget_labels']
    retain_labels = clustering_results['retain_labels']
    holdout_labels = clustering_results['holdout_labels']
    
    # Create figure
    if plot_dim == '3d':
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Colored by dataset type
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(forget_reduced[:, 0], forget_reduced[:, 1], forget_reduced[:, 2], 
                   c='red', label='Forget', alpha=0.6, s=50)
        ax1.scatter(retain_reduced[:, 0], retain_reduced[:, 1], retain_reduced[:, 2], 
                   c='blue', label='Retain', alpha=0.6, s=50)
        ax1.scatter(holdout_reduced[:, 0], holdout_reduced[:, 1], holdout_reduced[:, 2], 
                   c='green', label='Holdout', alpha=0.6, s=50)
        ax1.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax1.set_zlabel(f'PC3 ({explained_var[2]:.1%})')
        ax1.set_title('Data Split Visualization')
        ax1.legend()
        
        # Plot 2: Colored by cluster
        ax2 = fig.add_subplot(132, projection='3d')
        scatter = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                            c=cluster_labels, cmap='tab20', alpha=0.6, s=50)
        ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax2.set_zlabel(f'PC3 ({explained_var[2]:.1%})')
        ax2.set_title(f'HDBSCAN Clusters ({clustering_results["n_clusters"]} clusters)')
        plt.colorbar(scatter, ax=ax2, label='Cluster ID')
        
        # Plot 3: Combined view with dataset markers
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(forget_reduced[:, 0], forget_reduced[:, 1], forget_reduced[:, 2],
                   c=forget_labels, cmap='tab20', marker='o', alpha=0.6, s=50, label='Forget')
        ax3.scatter(retain_reduced[:, 0], retain_reduced[:, 1], retain_reduced[:, 2],
                   c=retain_labels, cmap='tab20', marker='^', alpha=0.6, s=50, label='Retain')
        ax3.scatter(holdout_reduced[:, 0], holdout_reduced[:, 1], holdout_reduced[:, 2],
                   c=holdout_labels, cmap='tab20', marker='s', alpha=0.6, s=50, label='Holdout')
        ax3.set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        ax3.set_zlabel(f'PC3 ({explained_var[2]:.1%})')
        ax3.set_title('Clusters by Dataset Type')
        ax3.legend()
        
    else:  # 2d
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Colored by dataset type
        axes[0].scatter(forget_reduced[:, 0], forget_reduced[:, 1], 
                       c='red', label='Forget', alpha=0.6, s=50)
        axes[0].scatter(retain_reduced[:, 0], retain_reduced[:, 1], 
                       c='blue', label='Retain', alpha=0.6, s=50)
        axes[0].scatter(holdout_reduced[:, 0], holdout_reduced[:, 1], 
                       c='green', label='Holdout', alpha=0.6, s=50)
        axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[0].set_title('Data Split Visualization')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Colored by cluster
        scatter = axes[1].scatter(reduced_data[:, 0], reduced_data[:, 1],
                                 c=cluster_labels, cmap='tab20', alpha=0.6, s=50)
        axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[1].set_title(f'HDBSCAN Clusters ({clustering_results["n_clusters"]} clusters)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1], label='Cluster ID')
        
        # Plot 3: Combined view with dataset markers
        axes[2].scatter(forget_reduced[:, 0], forget_reduced[:, 1],
                       c=forget_labels, cmap='tab20', marker='o', alpha=0.6, s=50, label='Forget')
        axes[2].scatter(retain_reduced[:, 0], retain_reduced[:, 1],
                       c=retain_labels, cmap='tab20', marker='^', alpha=0.6, s=50, label='Retain')
        axes[2].scatter(holdout_reduced[:, 0], holdout_reduced[:, 1],
                       c=holdout_labels, cmap='tab20', marker='s', alpha=0.6, s=50, label='Holdout')
        axes[2].set_xlabel(f'PC1 ({explained_var[0]:.1%})')
        axes[2].set_ylabel(f'PC2 ({explained_var[1]:.1%})')
        axes[2].set_title('Clusters by Dataset Type')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'HDBSCAN Clustering Results {title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def evaluate_model_on_benchmark(model_name, benchmark_name, subset_size=30, rephrasing_methods=['token_embedding_proximity'], dm=None):
    """
    Evaluate a single model on a benchmark using Input Loss Landscape (ILL) analysis
    
    Args:
        model_name: HuggingFace model name/path
        benchmark_name: One of 'TOFU', 'WMDP', 'MUSE' 
        subset_size: Number of examples per dataset split
        
    Returns:
        dict: Evaluation results including AUC scores and classification metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} on {benchmark_name}")
    print(f"{'='*60}")

        
    try:
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer_name = None
        if 'llama-2' in model_name.lower():
            tokenizer_name = "meta-llama/Llama-2-7b-hf"
        elif 'llama-3' in model_name.lower():
            tokenizer_name = "meta-llama/Meta-Llama-3-8B"
        elif 'zephyr' in model_name.lower():
            tokenizer_name = "TheBloke/Zephyr-7B-Beta-GPTQ"
            
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                    #   cache_dir=None,
                                                    )
        else:
            raise ValueError(f"Unknown tokenizer for model: {model_name}")
        
        try:    
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # cache_dir=None  # Disables caching; model is loaded directly into memory
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                # cache_dir=None  # Disables caching; model is loaded directly into memory
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.config.pad_token_id = tokenizer.eos_token_id
        
        # Set left padding only for decoder-only models (crucial for generation)
        if model.config.is_decoder and not getattr(model.config, 'is_encoder_decoder', False):
            tokenizer.padding_side = 'left'    
        
        print(f"Model loaded successfully on device: {next(model.parameters()).device}")
        
        
        
        # Prepare datasets for ILL evaluation
        
        if not REPHRASED_ORIGINAL:
            # Load benchmark data
            if "MUSE" in benchmark_name:
                data_loader = DATA_LOADERS[benchmark_name]
                forget_data, retain_data, holdout_data = data_loader(subset_size)
            else:
                data_loader = DATA_LOADERS[benchmark_name]
                forget_data, retain_data, holdout_data = data_loader(subset_size)
            # data_loader = DATA_LOADERS[benchmark_name]
            # forget_data, retain_data, holdout_data = data_loader(subset_size)
        
            datasets = {
                'forget': {'name': 'forget', 'data': forget_data},
                'retain': {'name': 'retain', 'data': retain_data}, 
                'holdout': {'name': 'holdout', 'data': holdout_data}
            }
            
            
            
            
        rephrasing_results = []
        
        LOSS_results = None  # To store LOSS-based results if computed
        
        for neighbor_method in tqdm(rephrasing_methods, total=len(rephrasing_methods), desc="Rephrasing Methods"):
            if REPHRASED_ORIGINAL:
                # use the matching rephrased data if available if not, assert that we only use the original
                strategy = STRATEGY.copy()
                strategy['method'] = neighbor_method

                forget_data = replace_original_with_rephrased(model_name, benchmark_name, 'forget', strategy, subset_size)
                retain_data = replace_original_with_rephrased(model_name, benchmark_name, 'retain', strategy, subset_size)
                holdout_data = replace_original_with_rephrased(model_name, benchmark_name, 'holdout', strategy, subset_size)
                datasets = {
                    'forget': {'name': 'forget', 'data': forget_data},
                    'retain': {'name': 'retain', 'data': retain_data}, 
                    'holdout': {'name': 'holdout', 'data': holdout_data}
                }
            try:
                print(f"\nEvaluating with rephrasing method: {neighbor_method}")
                results, binary_results, feature_importance_results, binary_feature_importance, trained_classifiers, clustering_results, features_dict = run_ill_evaluation(
                    model_name=model_name, 
                    benchmark_name=benchmark_name, 
                    model=model, 
                    tokenizer=tokenizer, 
                    datasets=datasets, 
                    neighbor_method=neighbor_method,
                    test_size=TEST_SIZE
                )
                
                # Save features_dict if dm is provided
                features_dict_metadata = {
                                            'model_name': model_name,
                                            'benchmark_name': benchmark_name,
                                        }
                # Ensure consistent sizes
                forget_tensor = features_dict['forget']['unnormalized_features_tensor']
                retain_tensor = features_dict['retain']['unnormalized_features_tensor']
                holdout_tensor = features_dict['holdout']['unnormalized_features_tensor']
                
                norm_forget_tensor, norm_retain_tensor, norm_holdout_tensor = eval_with_ILL.normalize_features(forget_tensor, retain_tensor, holdout_tensor)
                features_dict['forget']['normalized_features_tensor'] = norm_forget_tensor
                features_dict['retain']['normalized_features_tensor'] = norm_retain_tensor
                features_dict['holdout']['normalized_features_tensor'] = norm_holdout_tensor
                
                dm.save_tensors_and_metadata(features_dict, features_dict_metadata, filename_prefix="post_processed_features_dict")
                
                # Save clustering results if dm is provided
                if dm is not None:
                    clustering_metadata = {
                        'model_name': model_name,
                        'benchmark_name': benchmark_name,
                        'neighbor_method': neighbor_method,
                        'classifier_type': 'clustering',
                        'clf_name': 'hdbscan'
                    }
                    dm.save_classifier(clustering_results['clusterer'], clustering_metadata)
                    print(f"💾 Saved HDBSCAN clustering results")

                    fig_2d = clustering_results['figs']['2d']
                    fig_3d = clustering_results['figs']['3d']
                    dm.save_plot(fig_2d['fig'], fig_2d['name'])
                    dm.save_plot(fig_3d['fig'], fig_3d['name'])
                    print(f"💾 Saved HDBSCAN visualizations to:")
                    print(f"   - {fig_2d['name']}")
                    print(f"   - {fig_3d['name']}")
            

                # Add this: Save classifiers if dm is provided
                if dm is not None:
                    print(f"\n💾 Saving trained classifiers...")
                    saved_count = 0
                    for classifier_type in ['overall_predictors', 
                                            'binary_comparisons', 
                                            'retain_vs_all', 
                                            'forget_vs_all', 
                                            'holdout_vs_all',
                                            'retain_vs_forget',
                                            'retain_vs_holdout',
                                            'forget_vs_holdout'
                                            ]:
                        if classifier_type in trained_classifiers:
                            if classifier_type == 'binary_comparisons':
                                for clf_name, clf in trained_classifiers[classifier_type].items():
                                    if isinstance(clf, dict):
                                        for sub_clf_name, sub_clf in clf.items():
                                            metadata = {
                                                'model_name': model_name,
                                                'benchmark_name': benchmark_name,
                                                'neighbor_method': neighbor_method,
                                                'classifier_type': classifier_type,
                                                'clf_name': f"{clf_name}_{sub_clf_name}"
                                            }
                                            result = dm.save_classifier(sub_clf, metadata)
                                            if result:
                                                saved_count += 1
                                continue
                            for clf_name, clf in trained_classifiers[classifier_type].items():
                                metadata = {
                                    'model_name': model_name,
                                    'benchmark_name': benchmark_name,
                                    'neighbor_method': neighbor_method,
                                    'classifier_type': classifier_type,
                                    'clf_name': clf_name
                                }
                                result = dm.save_classifier(clf, metadata)
                                if result:
                                    saved_count += 1
                    print(f"✅ Successfully saved {saved_count} classifiers")

                    
                for classifier in ['logistic', 'random_forest']:
                    add_results = {
                        'neighbor_method': neighbor_method,
                        'classifier': classifier,
                        'Retain vs All AUC': binary_results['retain_vs_all'][classifier]['roc_auc'],
                        'Holdout vs All AUC': binary_results['holdout_vs_all'][classifier]['roc_auc'],
                        'Forget vs All AUC': binary_results['forget_vs_all'][classifier]['roc_auc'],
                        'Multi-class AUC': results['multi_class'][classifier]['roc_auc'],
                        'Retain_vs_All_auc_at_1_fp': binary_results['retain_vs_all'][classifier]['auc_at_1_fp'],
                        'Forget_vs_All_auc_at_1_fp': binary_results['forget_vs_all'][classifier]['auc_at_1_fp'],
                        'Holdout_vs_All_auc_at_1_fp': binary_results['holdout_vs_all'][classifier]['auc_at_1_fp'],
                        'Retain_vs_Forget_AUC': binary_results['retain_vs_forget'][classifier]['roc_auc'],
                        'Retain_vs_Holdout_AUC': binary_results['retain_vs_holdout'][classifier]['roc_auc'],
                        'Forget_vs_Holdout_AUC': binary_results['forget_vs_holdout'][classifier]['roc_auc'],
                        'Retain_vs_Forget_auc_at_1_fp': binary_results['retain_vs_forget'][classifier]['auc_at_1_fp'],
                        'Retain_vs_Holdout_auc_at_1_fp': binary_results['retain_vs_holdout'][classifier]['auc_at_1_fp'],
                        'Forget_vs_Holdout_auc_at_1_fp': binary_results['forget_vs_holdout'][classifier]['auc_at_1_fp'],
                        'binary_feature_importance': binary_feature_importance,
                        'feature_importance': feature_importance_results
                    }
                    rephrasing_results.append(add_results)
                    
                
                if LOSS_results is None:
                    # Store LOSS-based results only once
                    LOSS_results = binary_results.get('loss_based', None)
                    if LOSS_results:
                        print("Storing LOSS-based results.")
                    else:
                        print("No LOSS-based results found.")
                
                cleanup_gpu_memory()
            except Exception as e:
                print(f"❌ Error in rephrasing method {neighbor_method}: {e}")
                cleanup_gpu_memory()
                continue  # Proceed to next rephrasing method

        # Extract method name from model name
        method_name = "Unknown"
        for method_key, method_full in UNLEARNING_METHODS.items():
            if method_key in model_name.lower():
                method_name = method_full
                break
        if "simnpo" in model_name.lower():
            method_name = "SimNPO"
            
        prompt_type = identify_prompt_column_type(forget_data[0])
        forget_data = forget_data.map(lambda ex: filter_examples_by_num_tokens(ex, prompt_type, tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS))
        forget_data = forget_data.filter(lambda ex: ex['valid_example'])
        
        prompt_type = identify_prompt_column_type(retain_data[0])
        retain_data = retain_data.map(lambda ex: filter_examples_by_num_tokens(ex, prompt_type, tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS))
        retain_data = retain_data.filter(lambda ex: ex['valid_example'])

        prompt_type = identify_prompt_column_type(holdout_data[0])
        holdout_data = holdout_data.map(lambda ex: filter_examples_by_num_tokens(ex, prompt_type, tokenizer, max_prompt_tokens=MAX_PROMPT_TOKENS))
        holdout_data = holdout_data.filter(lambda ex: ex['valid_example'])

        # Compute baseline metrics
        print("Computing baseline metrics...")
        batch_sizes = [128, 64, 32, 16]  # Try these batch sizes in order
        
        
        def try_with_fallback_batch_sizes(func, *args, **kwargs):
            """Try a function with different batch sizes, falling back to smaller ones on failure."""
            if not COMPUTE_BASELINES:
                print("Skipping baseline computations as per configuration.")
                return {'retain_vs_all_auc': 0,
                        'forget_vs_all_auc': 0,
                        'holdout_vs_all_auc': 0,
                        'multi_class_auc': 0,
                        'retain_vs_all_auc_at_1_fp': 0,
                        'forget_vs_all_auc_at_1_fp': 0,
                        'holdout_vs_all_auc_at_1_fp': 0
                        }
            for bs in batch_sizes:
                try:
                    print(f"Trying {func.__name__} with batch_size={bs}...")
                    kwargs['batch_size'] = bs
                    result = func(*args, **kwargs)
                    print(f"✅ {func.__name__} succeeded with batch_size={bs}")
                    cleanup_gpu_memory()
                    return result
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f"❌ {func.__name__} failed with batch_size={bs}: {e}")
                    cleanup_gpu_memory()  # Clean up memory before retrying
                    continue
            # If all fail, raise the last error
            print(f"\033[93m⚠️ WARNING: All batch sizes failed for {func.__name__}\033[0m")
            return {'retain_vs_all_auc': 0,
                    'forget_vs_all_auc': 0,
                    'holdout_vs_all_auc': 0,
                    'multi_class_auc': 0,
                    'retain_vs_all_auc_at_1_fp': 0,
                    'forget_vs_all_auc_at_1_fp': 0,
                    'holdout_vs_all_auc_at_1_fp': 0
                    }
        
        cleanup_gpu_memory()
        
        if COMPUTE_BASELINES:
            print("Computing zlib compression AUC...")
            try:
                strategy = STRATEGY.copy()
                strategy['method'] = 'token_embedding_proximity'  # Use a default method for zlib baseline
                
                print(f"Loading generated neighbors for method: {neighbor_method}")
                forget_generated_data = load_neighbors_data(model_name=model_name, benchmark_name=benchmark_name, split_name='forget', strategy=strategy)
                retain_generated_data = load_neighbors_data(model_name=model_name, benchmark_name=benchmark_name, split_name='retain', strategy=strategy)
                holdout_generated_data = load_neighbors_data(model_name=model_name, benchmark_name=benchmark_name, split_name='holdout', strategy=strategy)
            
                zlib_results = compute_zlib_compression_auc(forget_generated_data, retain_generated_data, holdout_generated_data)
                
                cleanup_gpu_memory()
                
            except Exception as e:
                
                cleanup_gpu_memory()
                
                print(f"❌ Error computing zlib compression AUC: {e}")
                zlib_results = {'retain_vs_all_auc': 0,
                                'forget_vs_all_auc': 0,
                                'holdout_vs_all_auc': 0,
                                'multi_class_auc': 0,
                                'retain_vs_all_auc_at_1_fp': 0,
                                'forget_vs_all_auc_at_1_fp': 0,
                                'holdout_vs_all_auc_at_1_fp': 0
                                }
        else:
            zlib_results = {'retain_vs_all_auc': 0,
                            'forget_vs_all_auc': 0,
                            'holdout_vs_all_auc': 0,
                            'multi_class_auc': 0,
                            'retain_vs_all_auc_at_1_fp': 0,
                            'forget_vs_all_auc_at_1_fp': 0,
                            'holdout_vs_all_auc_at_1_fp': 0
                            }
            print("Skipping zlib compression AUC computation as per configuration.")
        
        min_k_results = try_with_fallback_batch_sizes( compute_min_k_auc,  forget_data, retain_data, holdout_data, model, tokenizer)
        # {'retain_vs_all_auc': 0,
        #                     'forget_vs_all_auc': 0,
        #                     'holdout_vs_all_auc': 0,
        #                     'multi_class_auc': 0,
        #                     'retain_vs_all_auc_at_1_fp': 0,
        #                     'forget_vs_all_auc_at_1_fp': 0,
        #                     'holdout_vs_all_auc_at_1_fp': 0
        #                     }
        
                
        min_k_pp_results = try_with_fallback_batch_sizes( compute_min_k_pp_auc,  forget_data, retain_data, holdout_data, model, tokenizer)
        
        rouge_results =  {'retain_vs_all_auc': 0,
                            'forget_vs_all_auc': 0,
                            'holdout_vs_all_auc': 0,
                            'multi_class_auc': 0,
                            'retain_vs_all_auc_at_1_fp': 0,
                            'forget_vs_all_auc_at_1_fp': 0,
                            'holdout_vs_all_auc_at_1_fp': 0
                            }
        # try_with_fallback_batch_sizes( compute_rouge_l_f1_auc,  forget_data, retain_data, holdout_data, model, tokenizer)
    
        cleanup_gpu_memory()

        # Compile results
        evaluation_results = {
            'model_name': model_name,
            'benchmark': benchmark_name,
            'method': method_name,
            'rephrasing_results': rephrasing_results,
            'binary_comparisons': binary_results,
            'feature_importance': feature_importance_results,
            'baselines': {
                            'zlib_compression': zlib_results,
                            'min_k_percent': min_k_results,
                            'min_k_pp_percent': min_k_pp_results,
                            'rouge_l_f1': rouge_results,
                            'loss_based': LOSS_results,
                            'spv_mia': binary_results.get('spv_mia', None)
                        },
        }
        
        print(f"✅ Evaluation completed for {model_name} on {benchmark_name}")
        # print(f"   Logistic Accuracy: {results['multi_class']['logistic']['accuracy']:.3f}")
        # print(f"   Random Forest Accuracy: {results['multi_class']['random_forest']['accuracy']:.3f}")
        
        # Clean up GPU memory
        del model, tokenizer
        # Clean up memory after each iteration
        cleanup_gpu_memory()
            
        return evaluation_results
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name} on {benchmark_name}: {e}")
        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def generate_neighbor_method_table(all_neighbor_results, data_manager):
    """
    Generate Table 5: REMIND_OURS Performance Across Neighbor Generation Methods
    
    Args:
        all_neighbor_results: Dict of results by method
        data_manager: DataManager instance
    
    Returns:
        pd.DataFrame: The new table
    """
    table_data = []
    
    for method, results in all_neighbor_results.items():
        if not results:
            continue
        
        # Aggregate AUC scores across results
        auc_scores = [r['multi_class']['logistic']['roc_auc'] for r in results if 'multi_class' in r]
        retain_auc_scores = [r['binary_comparisons'].get('retain_vs_all', {}).get('logistic', {}).get('roc_auc', 0.5) for r in results]
        forget_auc_scores = [r['binary_comparisons'].get('forget_vs_all', {}).get('logistic', {}).get('roc_auc', 0.5) for r in results]
        
        avg_auc = np.mean(auc_scores) if auc_scores else np.nan
        avg_retain_auc = np.mean(retain_auc_scores) if retain_auc_scores else np.nan
        avg_forget_auc = np.mean(forget_auc_scores) if forget_auc_scores else np.nan
        
        table_data.append({
            'Neighbor_Method': method,
            'Avg_Multi_Class_AUC': round(avg_auc, 3),
            'Avg_Retain_vs_All_AUC': round(avg_retain_auc, 3),
            'Avg_Forget_vs_All_AUC': round(avg_forget_auc, 3),
            'Num_Evaluations': len(results)
        })
    
    table5_df = pd.DataFrame(table_data)
    
    # Save via DataManager
    table5_path = os.path.join(data_manager.get_results_dir(), 'table5_neighbor_methods_comparison.csv')
    table5_df.to_csv(table5_path, index=False)
    print(f"📊 Table 5 saved to: {table5_path}")
    
    return table5_df


#create a class for data manager for saving/loading results
import os
import json
import pandas as pd
from datetime import datetime

class DataManager:
    """
    A class to manage saving and loading of experiment results and analysis data.
    
    Handles CSV, JSON, and other file formats for results storage.
    """
    
    def __init__(self, base_dir=None, experiment_name=None, TIMESTAMP=None, save_plots=True):
        """
        Initialize the DataManager.
        
        Args:
            base_dir: Base directory for results (default: current working directory)
            experiment_name: Name for the experiment (default: timestamp-based)
        """
        if base_dir is None:
            base_dir = os.getcwd()
            
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.save_plots = save_plots

        if experiment_name is None:
            experiment_name = f"experiment_{TIMESTAMP}"
        else:
            experiment_name = f"{experiment_name}_{TIMESTAMP}"

        self.results_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)
        self.classifiers = {}
        
        print(f"DataManager initialized. Results will be saved to: {self.results_dir}")
        
    def set_results_dir(self, experiment_name):
        """
        Set a new results directory.
        
        Args:
            experiment_name: New experiment name
        """
        experiment_name = f"{experiment_name}_{TIMESTAMP}"
        self.results_dir = os.path.join(self.base_dir, f"{experiment_name}")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results directory updated to: {self.results_dir}")
        
    def set_subdirectory(self, subdirectory_name):
        """
        Set a subdirectory within the results directory.
        
        Args:
            subdirectory_name: Name of the subdirectory
        """
        subdirectory_name = subdirectory_name.strip().replace(' ', '_').replace('/', '_')
        self.results_dir = os.path.join(self.results_dir, subdirectory_name)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results subdirectory updated to: {self.results_dir}")
    
    def return_one_subdir_up(self):
        self.results_dir = os.path.dirname(self.results_dir)
        print(f"Returned one subdirectory up. Current results directory: {self.results_dir}")
    
    def save_experiment_results(self, all_results, filename='experiment_results.json'):
        """
        Save the complete experiment results as JSON.
        
        Args:
            all_results: List of evaluation results
            filename: Name of the file to save to
        """
        filepath = os.path.join(self.results_dir, filename)
        
        def make_json_serializable(obj):
            if isinstance(obj, (dict)):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_results = [make_json_serializable(result) for result in all_results]
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Experiment results saved to: {filepath}")
        return filepath
    
    def save_analysis_results(self, analysis_results, filename_prefix='analysis'):
        """
        Save analysis results including tables and statistics.
        
        Args:
            analysis_results: Dictionary containing tables and stats
            filename_prefix: Prefix for filenames
        """
        saved_files = {}
        
            # Log tables to W&B
        def df_to_wandb_table(df, table_name):
            """Helper to convert DataFrame to W&B Table."""
            if df.empty:
                return None
            table = wandb.Table(dataframe=df)
            wandb.log({table_name: table})
            return table
        
        # Save tables as CSV
        if 'table2' in analysis_results:
            table2_path = os.path.join(self.results_dir, f'{filename_prefix}_table2.csv')
            analysis_results['table2'].to_csv(table2_path, index=False)
            saved_files['table2'] = table2_path
            print(f"Table 2 saved to: {table2_path}")
            
            df_to_wandb_table(analysis_results['table2'], "table2_aggregate_comparison")
        
        if 'table3' in analysis_results:
            table3_path = os.path.join(self.results_dir, f'{filename_prefix}_table3.csv')
            analysis_results['table3'].to_csv(table3_path)
            saved_files['table3'] = table3_path
            print(f"Table 3 saved to: {table3_path}")
            
            df_to_wandb_table(analysis_results['table3'], "table3_model_family_comparison")
        
        if 'table4' in analysis_results:
            table4_path = os.path.join(self.results_dir, f'{filename_prefix}_table4.csv')
            analysis_results['table4'].to_csv(table4_path, index=False)
            saved_files['table4'] = table4_path
            print(f"Table 4 saved to: {table4_path}")
            
            df_to_wandb_table(analysis_results['table4'], "table4_detailed_results")
        
        if 'detailed_results' in analysis_results:
            detailed_path = os.path.join(self.results_dir, f'{filename_prefix}_detailed_results.csv')
            analysis_results['detailed_results'].to_csv(detailed_path, index=False)
            saved_files['detailed_results'] = detailed_path
            print(f"Detailed results saved to: {detailed_path}")
            
            df_to_wandb_table(analysis_results['detailed_results'], "detailed_results_full")
        
        # Save summary stats as JSON
        if 'summary_stats' in analysis_results:
            stats_path = os.path.join(self.results_dir, f'{filename_prefix}_summary.json')
            with open(stats_path, 'w') as f:
                json.dump(analysis_results['summary_stats'], f, indent=2)
            saved_files['summary'] = stats_path
            print(f"Summary statistics saved to: {stats_path}")
        
        return saved_files
    
    def save_to_csv(self, all_results, prefix_name='intermediate'):
        """
        Save intermediate results during long-running experiments.
        
        Args:
            all_results: Current list of results
            prefix_name: Identifier for this intermediate save
        """
        filename = f'{prefix_name}.csv'
        filepath = os.path.join(self.results_dir, filename)
        
        
        if not isinstance(all_results, pd.DataFrame):
            # Convert to DataFrame for easy CSV saving
            all_results = pd.DataFrame(all_results)
        all_results.to_csv(filepath, index=False)
        
        print(f"Intermediate results saved to: {filepath}")
        return filepath
    
    def load_experiment_results(self, filename='experiment_results.json'):
        """
        Load previously saved experiment results.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            List of experiment results
        """
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        print(f"Experiment results loaded from: {filepath}")
        return results
    
    def load_analysis_table(self, table_name, filename_prefix='analysis'):
        """
        Load a specific analysis table.
        
        Args:
            table_name: 'table2', 'table3', 'table4', or 'detailed_results'
            filename_prefix: Prefix used when saving
            
        Returns:
            pandas.DataFrame
        """
        filename = f'{filename_prefix}_{table_name}.csv'
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Table file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Table {table_name} loaded from: {filepath}")
        return df
    
    def load_summary_stats(self, filename_prefix='analysis'):
        """
        Load summary statistics.
        
        Args:
            filename_prefix: Prefix used when saving
            
        Returns:
            Dictionary of summary statistics
        """
        filename = f'{filename_prefix}_summary.json'
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Summary file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            stats = json.load(f)
        
        print(f"Summary statistics loaded from: {filepath}")
        return stats
    
    def list_saved_files(self):
        """
        List all files in the results directory.
        
        Returns:
            List of filenames
        """
        if not os.path.exists(self.results_dir):
            return []
        
        files = os.listdir(self.results_dir)
        print(f"Files in results directory ({self.results_dir}):")
        for file in files:
            print(f"  - {file}")
        
        return files
    
    def load_individual_result(self, model_name, benchmark_name):
        """
        Load results for a specific model-benchmark combination.
        
        Args:
            model_name: HuggingFace model name/path
            benchmark_name: Benchmark name (TOFU, WMDP, MUSE)
            
        Returns:
            Dictionary of results or None if not found
        """
        # Create safe filename for this model-benchmark combination
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        result_filename = f"result_{safe_model_name}_{benchmark_name}.json"
        result_filepath = os.path.join(self.results_dir, result_filename)
        
        if not os.path.exists(result_filepath):
            print(f"Result not found: {result_filename}")
            return None
        
        try:
            with open(result_filepath, 'r') as f:
                result = json.load(f)
            print(f"Individual result loaded from: {result_filename}")
            return result
        except Exception as e:
            print(f"Error loading result from {result_filename}: {e}")
            return None
    
    def load_all_individual_results(self):
        """
        Load all individual model results from the results directory.
        
        Returns:
            List of all individual results
        """
        all_results = []
        
        if not os.path.exists(self.results_dir):
            print(f"Results directory does not exist: {self.results_dir}")
            return all_results
        
        # Find all result files
        result_files = [f for f in os.listdir(self.results_dir) if f.startswith('result_') and f.endswith('.json')]
        
        print(f"Found {len(result_files)} individual result files")
        
        for result_file in result_files:
            result_filepath = os.path.join(self.results_dir, result_file)
            try:
                with open(result_filepath, 'r') as f:
                    result = json.load(f)
                all_results.append(result)
                print(f"  ✅ Loaded: {result_file}")
            except Exception as e:
                print(f"  ❌ Failed to load {result_file}: {e}")
        
        print(f"Successfully loaded {len(all_results)} individual results")
        return all_results
    
    def save_tensors(self, tensors_dict, filename_prefix='tensors', to_base_dir=False):
        """
        Save tensors to files.
        """
        def convert_tensors(obj):
            if isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_tensors(v) for v in obj]
            elif isinstance(obj, torch.Tensor) and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                return obj.cpu().numpy()
            else:
                return obj

        # Convert tensors to numpy for serialization
        serializable_features = convert_tensors(tensors_dict)
        
        filename = os.path.join(self.base_dir if to_base_dir else self.results_dir, f'{filename_prefix}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(serializable_features, f)
            
    def load_tensors(self, filename_prefix='tensors', from_base_dir=False):
        """
        Load tensors from files.
        """
        filename = os.path.join(self.base_dir if from_base_dir else self.results_dir, f'{filename_prefix}.pkl')
        with open(filename, 'rb') as f:
            tensors = pickle.load(f)
        return tensors
        
    def save_metadata(self, metadata, filename_prefix='visualization', to_base_dir=False):
        """
        Save visualization metadata and file paths.
        
        Args:
            metadata: Data to be saved as metadata. Accepts any JSON-serializable data
            filename_prefix: Prefix for the metadata filename
            to_base_dir: If True, save metadata to the base directory instead of the results directory
            
        Returns:
            str: Path to saved metadata file
        """
        
        metadata_path = os.path.join(self.base_dir if to_base_dir else self.results_dir, f'{filename_prefix}_metadata.json')
        
        def make_json_serializable(obj):
            if isinstance(obj, (dict)):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_data = make_json_serializable(metadata)
        
        with open(metadata_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Visualization metadata saved to: {metadata_path}")
        return metadata_path
    
    def load_metadata(self, filename_prefix='visualization', from_base_dir=False):
        """
        Load visualization metadata.
        
        Args:
            filename_prefix: Prefix used when saving
            from_base_dir: If True, load metadata from the base directory instead of the results directory
            
        Returns:
            Dictionary of visualization metadata
        """
        metadata_path = os.path.join(self.base_dir if from_base_dir else self.results_dir, f'{filename_prefix}_metadata.json')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Visualization metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Visualization metadata loaded from: {metadata_path}")
        return metadata
    
    def list_visualizations(self):
        """
        List all visualization files in the results directory.
        
        Returns:
            Dictionary with visualization types and their file paths
        """
        vis_files = {}
        
        if not os.path.exists(self.results_dir):
            print(f"Results directory does not exist: {self.results_dir}")
            return vis_files
        
        # Common visualization file patterns
        patterns = {
            'overview': ['performance_overview', 'overview'],
            'method_comparison': ['method_comparison', 'method'],
            'benchmark_comparison': ['benchmark_comparison', 'benchmark'],
            'classifier_comparison': ['classifier_comparison', 'classifier'],
            'individual_plots': ['individual_', 'detail_']
        }
        
        all_files = os.listdir(self.results_dir)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg'))]
        
        print(f"Found {len(image_files)} visualization files:")
        
        for file in image_files:
            file_path = os.path.join(self.results_dir, file)
            
            # Categorize files based on patterns
            categorized = False
            for category, pattern_list in patterns.items():
                if any(pattern in file.lower() for pattern in pattern_list):
                    if category not in vis_files:
                        vis_files[category] = []
                    vis_files[category].append(file_path)
                    categorized = True
                    break
            
            if not categorized:
                if 'other' not in vis_files:
                    vis_files['other'] = []
                vis_files['other'].append(file_path)
            
            print(f"  📊 {file}")
        
        return vis_files
    
    def load_specific_visualization(self, visualization_name):
        """
        Load a specific visualization by name pattern.
        
        Args:
            visualization_name: Name pattern to search for (e.g., 'overview', 'method')
            
        Returns:
            List of file paths matching the pattern
        """
        vis_files = self.list_visualizations()
        
        matching_files = []
        for category, files in vis_files.items():
            if visualization_name.lower() in category.lower():
                matching_files.extend(files)
        
        # Also check individual file names
        if not matching_files:
            all_files = os.listdir(self.results_dir)
            for file in all_files:
                if (visualization_name.lower() in file.lower() and 
                    file.lower().endswith(('.png', '.pdf', '.svg', '.jpg', '.jpeg'))):
                    matching_files.append(os.path.join(self.results_dir, file))
        
        if matching_files:
            print(f"Found {len(matching_files)} visualization(s) matching '{visualization_name}':")
            for file in matching_files:
                print(f"  📊 {os.path.basename(file)}")
        else:
            print(f"No visualizations found matching '{visualization_name}'")
        
        return matching_files
    
    def save_individual_result(self, result, model_name, benchmark_name):
        """
        Save an individual evaluation result as JSON.
        
        Args:
            result: Dictionary containing the evaluation result
            model_name: HuggingFace model name/path
            benchmark_name: Benchmark name (TOFU, WMDP, MUSE)
            
        Returns:
            str: Path to the saved file
        """
        # Create safe filename
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        result_filename = f"result_{safe_model_name}_{benchmark_name}.json"
        result_filepath = os.path.join(self.results_dir, result_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(result_filepath), exist_ok=True)
        
        # Make result JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, (dict)):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_result = make_json_serializable(result)
        
        try:
            with open(result_filepath, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            print(f"💾 Individual result saved to: {result_filename}")
            return result_filepath
        except Exception as e:
            print(f"⚠️ Failed to save individual result: {e}")
            return None
    
    def get_results_dir(self):
        """
        Get the results directory path.
        
        Returns:
            str: Path to the results directory
        """
        return self.results_dir
    
    def save_plot(self, fig, filename):
        """Save plot if save_plots is True"""
        if self.save_plots and isinstance(fig, plt.Figure):
                filepath = os.path.join(self.results_dir, f"{filename}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved plot: {filepath}")
        elif self.save_plots:
            raise ValueError("Provided fig is not a matplotlib Figure instance.")
    
    def save_classifier(self, classifier, metadata):
        """
        Save a trained sklearn classifier using joblib.
        
        Args:
            classifier: Trained sklearn model
            filename: Name of the file (e.g., 'logistic_classifier.pkl')
            metadata: Additional metadata to save with the classifier
        """
        # if DEBUG:
        #     print("Debug mode: skipping saving classifier")
        #     return None
        if not SAVE_CLASSIFIERS:
            print("Saving classifiers is disabled by configuration.")
            return None
        try:
            assert isinstance(metadata, dict), "Metadata must be a dictionary"
            model_name = metadata.get("model_name", "unknown_model")
            benchmark_name = metadata.get("benchmark_name", "unknown_benchmark")
            neighbor_method = metadata.get("neighbor_method", "unknown_method")
            classifier_type = metadata.get("classifier_type", "unknown_type")
            clf_name = metadata.get("clf_name", "unknown_clf")
            filename = f"{model_name.replace('/', '_')}_{benchmark_name}_{neighbor_method}_{classifier_type}_{clf_name}_classifier.pkl"
            filepath = os.path.join(self.results_dir, filename)
            joblib.dump(classifier, filepath)
            print(f"💾 Classifier saved to: {filepath}")
            metadata['filepath'] = filepath
            self.classifiers[filename] = metadata
            return filepath
        except Exception as e:
            print(f"❌ Error saving classifier {filename}: {e}")
            return None
    
    def load_classifier(self, filename):
        """
        Load a trained sklearn classifier using joblib.
        
        Args:
            filename: Name of the file (e.g., 'logistic_classifier.pkl')
            
        Returns:
            Loaded classifier model
        """
        try:
            metadata = self.classifiers.get(filename, {})
            if metadata == {}:
                print(f"No metadata found for classifier: {filename}")
                filepath = os.path.join(self.results_dir, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Classifier file not found: {filepath}")
                classifier = joblib.load(filepath)
                print(f"📂 Classifier loaded from: {filepath}")
                return classifier, None
            
            filepath = metadata.get('filepath', os.path.join(self.results_dir, filename))
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Classifier file not found: {filepath}")
            classifier = joblib.load(filepath)
            print(f"📂 Classifier loaded from: {filepath}")
            return classifier, metadata            
        except Exception as e:
            print(f"❌ Error loading classifier {filename}: {e}")
            return None
    
    def save_list_classifiers(self, filename_prefix="classifiers"):
        """
        Save the list of classifiers to a file.
        """
        if SAVE_CLASSIFIERS:
            self.save_metadata(metadata = self.classifiers, filename_prefix=filename_prefix, to_base_dir=True)
        
    def get_list_classifiers(self, filename_prefix="classifiers"):
        """
        List all saved classifier files in the results directory.
        
        Returns:
            classifier_files = List of classifier filenames
            metadata = Dictionary containing metadata for each classifier or None if not available
        """
        classifier_files, metadata = [], None
        if self.classifiers:
            classifier_files = list(self.classifiers.keys())
            return classifier_files, self.classifiers
        
        elif not filename_prefix.endswith('.json'):
            classifiers_metadata = self.load_metadata(filename_prefix=filename_prefix, from_base_dir=True)
            classifier_files = list(classifiers_metadata.keys())
            return classifier_files, classifiers_metadata
            
        elif filename_prefix.endswith('.json'):
            filename = filename_prefix
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    classifiers_metadata = json.load(f)
                    
                classifier_files = list(classifiers_metadata.keys())
                return classifier_files, classifiers_metadata
            else:
                print(f"Metadata file not found in: {filename}")
                return {}
        else:
            print(f"No classifiers have been saved yet. Searching in {self.results_dir} for *_classifier.pkl files")
            
            if not os.path.exists(self.results_dir):
                print(f"Results directory does not exist: {self.results_dir}")
                return []
            
            classifier_files = [f for f in os.listdir(self.results_dir) if f.endswith('_classifier.pkl')]
            
            if classifier_files:
                print(f"Found {len(classifier_files)} classifier files:")
                for file in classifier_files:
                    print(f"  🤖 {file}")
            else:
                print("No classifier files found.")
            
        if metadata is None:
            print("No metadata available for classifiers.")
        if not classifier_files:
            print("No classifier files found.")
        return classifier_files, metadata
    
    def save_tensors_and_metadata(self, features_dict, metadata, filename_prefix):
        """
        Save normalized feature tensors and metadata using DataManager.
        
        Args:
            data_manager: DataManager instance
            features_dict: Dictionary containing feature tensors for forget, retain, holdout splits
            metadata: Dictionary containing metadata (model_name, benchmark_name, neighbor_method, etc.)
            filename_prefix: Prefix for the filename to save tensors and metadata
        """  
        # Create a unique base filename from metadata
        model_name = metadata.get('model_name', 'unknown').replace('/', '_')
        benchmark_name = metadata.get('benchmark_name', 'unknown')
        
        self.set_subdirectory(f"{model_name}_{benchmark_name}")
        
        self.save_tensors(tensors_dict=features_dict, filename_prefix=filename_prefix)
        
        self.return_one_subdir_up()
        
    def load_tensors_and_metadata(self, metadata, filename_prefix):
        """
        Load normalized feature tensors and metadata using DataManager.
        
        Args:
            data_manager: DataManager instance
            filename_prefix: Prefix for the filename to load tensors and metadata
        Returns:
            features_dict: Dictionary containing feature tensors for forget, retain, holdout splits
        """
        
        model_name = metadata.get('model_name', 'unknown').replace('/', '_')
        benchmark_name = metadata.get('benchmark_name', 'unknown')
        self.set_subdirectory(f"{model_name}_{benchmark_name}")
        
        features_dict = self.load_tensors(filename_prefix=filename_prefix)
        
        self.return_one_subdir_up()    


def run_comprehensive_experiments(max_models_per_family=3, subset_size=25, force_rerun=False, data_manager=None, rephrasing_methods=['token_embedding_proximity', 'random_token_replacement', 'context_based_token_replacement']):
    """
    Run comprehensive experiments across all model families and benchmarks, testing each with multiple neighbor generation methods.
    
    Args:
        max_models_per_family: Maximum number of models to test per family (for time constraints)
        subset_size: Number of examples per dataset split
        force_rerun: If True, rerun experiments even if results already exist
        data_manager: DataManager instance for saving/loading results
        neighbor_methods: List of neighbor generation methods to test (e.g., ['token_embedding_proximity', 'random_token_replacement', 'context_based_token_replacement', 'rephrasing'])
        
    Returns:
        list: All evaluation results (one per model-benchmark-method combination)
    """
    print(f"\n🚀 Starting comprehensive experiments...")
    print(f"   Max models per family: {max_models_per_family}")
    print(f"   Subset size: {subset_size}")
    print(f"   Force rerun: {force_rerun}")
    print(f"   Neighbor methods: {rephrasing_methods}")
    
    if data_manager is None:
        print("⚠️ No DataManager provided, creating default one")
        data_manager = DataManager(base_dir=os.path.join(Unlearn_Simple_DIR, 'results'), 
                                 experiment_name="unlearning_evaluation")
    
    cleanup_gpu_memory()
    
    all_results = []
    
    # Define model-benchmark mappings
    model_benchmark_mapping = {
        # WMDP models -> WMDP benchmark
        **{model: "WMDP" for model in MODEL_CONFIGS["wmdp_models"][:max_models_per_family]},
        **{model: "WMDP" for model in MODEL_CONFIGS["wmdp_zephyr"][:max_models_per_family]},
        
        # TOFU models -> TOFU benchmark  
        **{model: "TOFU" for model in MODEL_CONFIGS["tofu_models"][:max_models_per_family]},
        
        # MUSE models -> MUSE benchmark
        "OPTML-Group/SimNPO-MUSE-News-llama-2-7b": "MUSE_NEWS",
        "OPTML-Group/SimNPO-MUSE-Books-llama-2-7b": "MUSE_BOOKS",
    }
    
    total_combinations = len(model_benchmark_mapping) * len(rephrasing_methods)
    print(f"Total model-benchmark combinations: {len(model_benchmark_mapping)}")
    print(f"Total experiments (with neighbor methods): {total_combinations}")
    
    # Check for existing results
    skipped_count = 0
    loaded_count = 0
    
    # Run evaluations
    for i, (model_name, benchmark_name) in enumerate(model_benchmark_mapping.items(), 1):
        
         # Debug mode: limit to first 2 evaluations
        if DEBUG and i >= 2:
            print("Debug mode: stopping after 2 evaluations")
            break
        print(f"   Model: {model_name}")
        print(f"   Benchmark: {benchmark_name}")
        
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        result_filename = f"result_{safe_model_name}_{benchmark_name}.json"
        result_filepath = os.path.join(data_manager.get_results_dir(), result_filename)
        
        # Check if result already exists
        if not force_rerun and os.path.exists(result_filepath):
            print(f"✅ Found existing result, loading from: {result_filename}")
            try:
                with open(result_filepath, 'r') as f:
                    result = json.load(f)
                all_results.append(result)
                loaded_count += 1
                continue
            except Exception as e:
                print(f"⚠️ Failed to load existing result: {e}, running experiment...")
        
        try:
            result = evaluate_model_on_benchmark(model_name, benchmark_name, subset_size, rephrasing_methods=rephrasing_methods, dm=dm)
        except Exception as e:
                print(f"❌ Exception during evaluation: {e}")
                result = None
                raise e
                
        if result is not None:
            # Add neighbor method to result for tracking
            all_results.append(result)
            
            # Save individual result
            # In the run_comprehensive_experiments function, add this line before saving individual results to ensure the directory exists

            result_filepath = data_manager.save_individual_result(result, model_name, benchmark_name)
        else:
            print(f"❌ Evaluation failed for {model_name} on {benchmark_name}")
        
        # Clean up memory after each iteration
        cleanup_gpu_memory()
    
    print(f"\n🎉 Comprehensive experiments completed!")
    print(f"   Successful evaluations: {len(all_results)}/{total_combinations}")
    print(f"   Loaded from cache: {loaded_count}")
    print(f"   Newly evaluated: {len(all_results) - loaded_count}")
    
    # Save final combined results
    data_manager.save_list_classifiers()
    data_manager.save_experiment_results(all_results, filename='all_experiment_results.json')
    
    return all_results


def create_comprehensive_detailed_results(all_results, save_path=None):
    """
    Create a comprehensive detailed results DataFrame that includes ALL methods:
    - REMIND_OURS (our method) with different neighbor methods and classifiers
    - All baseline methods (Zlib, MIN-K%, MIN-K%++, ROUGE-L F1) if available
    
    Args:
        all_results: List of evaluation results from experiments
        save_path: Optional path to save the results CSV
        
    Returns:
        pandas.DataFrame: Comprehensive detailed results
    """
    print(f"\n📋 Creating comprehensive detailed results from {len(all_results)} evaluations...")
    
    detailed_all_results = []
    
    for result in all_results:
        model_name = result['model_name'].split('/')[-1]
        benchmark = result['benchmark']
        method = result['method']
        
        # 1. Add REMIND_OURS results for each rephrasing method and classifier
        for rephrasing in result.get('rephrasing_results', []):
            detailed_all_results.append({
                'Model': model_name,
                'Benchmark': benchmark,
                'Unlearning_Method': method,
                'Evaluation_Method': 'REMIND_OURS',
                'Neighbor_Method': rephrasing.get('neighbor_method', 'Unknown'),
                'Classifier': rephrasing.get('classifier', 'Unknown'),
                'Multi_class_AUC': rephrasing.get('Multi-class AUC', np.nan),
                'Accuracy': np.nan,  # Not available in rephrasing results
                'F1': np.nan,  # Not available in rephrasing results
                'Retain_vs_All_AUC': rephrasing.get('Retain vs All AUC', np.nan),
                'Forget_vs_All_AUC': rephrasing.get('Forget vs All AUC', np.nan),
                'Holdout_vs_All_AUC': rephrasing.get('Holdout vs All AUC', np.nan),
                'Retain_vs_Forget_AUC': rephrasing.get('Retain_vs_Forget_AUC', np.nan),
                'Retain_vs_Holdout_AUC': rephrasing.get('Retain_vs_Holdout_AUC', np.nan),
                'Forget_vs_Holdout_AUC': rephrasing.get('Forget_vs_Holdout_AUC', np.nan),
                'AUC_at_1_FP': rephrasing.get('Retain_vs_All_auc_at_1_fp', np.nan),  # Using Retain as representative
                'Method_Category': 'Our Method'
            })
        
        # 2. Add baseline method results if available
        baselines = result.get('baselines', {})
        if baselines:  # Check if baselines exist and are not empty
            baseline_mapping = {
                'zlib_compression': 'Zlib Compression',
                'min_k_percent': 'MIN-K%',
                'min_k_pp_percent': 'MIN-K%++',
                'rouge_l_f1': 'ROUGE-L F1'
            }
            
            for baseline_key, baseline_name in baseline_mapping.items():
                if baseline_key in baselines:
                    baseline_data = baselines[baseline_key]
                    # Only process if baseline_data is a dictionary (i.e., results were computed)
                    if isinstance(baseline_data, dict):
                        detailed_all_results.append({
                            'Model': model_name,
                            'Benchmark': benchmark,
                            'Unlearning_Method': method,
                            'Evaluation_Method': baseline_name,
                            'Neighbor_Method': 'N/A',  # Baselines don't have neighbor methods
                            'Classifier': 'N/A',  # Baselines don't use classifiers
                            'Multi_class_AUC': baseline_data.get('multi_class_auc', 0.5),
                            'Accuracy': np.nan,
                            'F1': np.nan,
                            'Retain_vs_All_AUC': baseline_data.get('retain_vs_all_auc', 0.5),
                            'Forget_vs_All_AUC': baseline_data.get('forget_vs_all_auc', 0.5),
                            'Holdout_vs_All_AUC': baseline_data.get('holdout_vs_all_auc', 0.5),
                            'Retain_vs_Forget_AUC': np.nan,  # Not available for baselines
                            'Retain_vs_Holdout_AUC': np.nan,  # Not available for baselines
                            'Forget_vs_Holdout_AUC': np.nan,  # Not available for baselines
                            'AUC_at_1_FP': baseline_data.get('retain_vs_all_auc_at_1_fp', 0.5),
                            'Method_Category': 'Baseline'
                        })
                    # If baseline_data is not a dict (e.g., 0), skip it (treat as not computed)
    
    # Convert to DataFrame
    detailed_all_df = pd.DataFrame(detailed_all_results)
    
    # Add model family information
    def get_model_family(model_name):
        if 'llama-3' in model_name.lower():
            return 'LLaMA-3-8B'
        elif 'llama-2' in model_name.lower():
            return 'LLaMA-2-7B'
        elif 'zephyr' in model_name.lower():
            return 'Zephyr-7B'
        else:
            return 'Other'
    
    detailed_all_df['Model_Family'] = detailed_all_df['Model'].apply(get_model_family)
    
    # Reorder columns for better readability
    column_order = [
        'Model', 'Model_Family', 'Benchmark', 'Unlearning_Method', 
        'Evaluation_Method', 'Neighbor_Method', 'Classifier', 'Method_Category',
        'Multi_class_AUC', 'Retain_vs_All_AUC', 'Forget_vs_All_AUC', 
        'Holdout_vs_All_AUC', 'Retain_vs_Forget_AUC', 'Retain_vs_Holdout_AUC', 
        'Forget_vs_Holdout_AUC', 'AUC_at_1_FP', 'Accuracy', 'F1'
    ]
    detailed_all_df = detailed_all_df[column_order]
    
    # Sort by benchmark, model, and evaluation method
    detailed_all_df = detailed_all_df.sort_values([
        'Benchmark', 'Model_Family', 'Model', 'Evaluation_Method', 'Neighbor_Method', 'Classifier'
    ]).reset_index(drop=True)
    
    print(f"✅ Created comprehensive detailed results:")
    print(f"   Total rows: {len(detailed_all_df)}")
    print(f"   REMIND_OURS results: {len(detailed_all_df[detailed_all_df['Method_Category'] == 'Our Method'])}")
    print(f"   Baseline results: {len(detailed_all_df[detailed_all_df['Method_Category'] == 'Baseline'])}")
    print(f"   Unique models: {detailed_all_df['Model'].nunique()}")
    print(f"   Unique benchmarks: {detailed_all_df['Benchmark'].nunique()}")
    print(f"   Evaluation methods: {list(detailed_all_df['Evaluation_Method'].unique())}")
    
    # Save if path provided
    if save_path:
        detailed_all_df.to_csv(save_path, index=False)
        print(f"📁 Detailed all results saved to: {save_path}")
    
    return detailed_all_df


def display_detailed_results_summary(detailed_all_df):
    """
    Display a summary of the comprehensive detailed results
    
    Args:
        detailed_all_df: DataFrame from create_comprehensive_detailed_results()
    """
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE DETAILED RESULTS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   Total evaluations: {len(detailed_all_df)}")
    print(f"   Unique models: {detailed_all_df['Model'].nunique()}")
    print(f"   Model families: {list(detailed_all_df['Model_Family'].unique())}")
    print(f"   Benchmarks: {list(detailed_all_df['Benchmark'].unique())}")
    print(f"   Evaluation methods: {list(detailed_all_df['Evaluation_Method'].unique())}")
    
    # Method breakdown
    print(f"\n🔍 Method Breakdown:")
    method_counts = detailed_all_df['Evaluation_Method'].value_counts()
    for method, count in method_counts.items():
        category = detailed_all_df[detailed_all_df['Evaluation_Method'] == method]['Method_Category'].iloc[0]
        print(f"   {method} ({category}): {count} evaluations")
    
    # Performance summary by evaluation method
    print(f"\n🏆 Performance Summary (Mean Multi-class AUC):")
    performance_summary = detailed_all_df.groupby(['Evaluation_Method', 'Classifier'])['Multi_class_AUC'].agg(['mean', 'std', 'count']).round(3)
    
    for (method, classifier), stats in performance_summary.iterrows():
        if classifier != 'N/A':
            print(f"   {method} ({classifier}): {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
        else:
            print(f"   {method}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Best performing combinations
    print(f"\n🥇 Top 10 Best Performing Combinations:")
    top_10 = detailed_all_df.nlargest(10, 'Multi_class_AUC')[
        ['Model', 'Benchmark', 'Evaluation_Method', 'Classifier', 'Multi_class_AUC']
    ]
    
    for idx, row in top_10.iterrows():
        classifier_str = f" ({row['Classifier']})" if row['Classifier'] != 'N/A' else ""
        print(f"   {row['Model']} | {row['Benchmark']} | {row['Evaluation_Method']}{classifier_str}: {row['Multi_class_AUC']:.3f}")
    
    print("\n" + "="*80)

print("✅ Comprehensive detailed results functions defined!")


dm = DataManager(base_dir=RESULTS_DIR, experiment_name=EXPERIMENT_NAME, TIMESTAMP=TIMESTAMP)


# assert not DEBUG, "Set DEBUG = False to run full experiments"


# Run the comprehensive experiments
print("🚀 Starting comprehensive evaluation across all models and benchmarks...")
print("This may take some time depending on available computational resources.")

# Configure experiment parameters
MAX_MODELS_PER_FAMILY = 5  # Limit for computational efficiency

print(f"Configuration:")
print(f"  Max models per family: {MAX_MODELS_PER_FAMILY}")
print(f"  Dataset subset size: {SUBSET_SIZE}")
print(f"  Expected total evaluations: ~{MAX_MODELS_PER_FAMILY * 3 * 2}")  # families * benchmarks * classifiers

# Run experiments with DataManager
all_results = run_comprehensive_experiments(
    max_models_per_family=MAX_MODELS_PER_FAMILY,
    subset_size=SUBSET_SIZE,
    force_rerun= True,  # Set to True to rerun all experiments
    data_manager=dm,
    rephrasing_methods=['token_embedding_proximity']
)
all_results


# Alternative: Load existing results if experiments were already run
print("💡 Alternative options:")
print("   1. Run with force_rerun=True to rerun all experiments")
print("   2. Load existing individual results from DataManager")
print("   3. Run a subset of models for testing")

# Example: Load existing results if available
existing_results = dm.load_all_individual_results()
if len(existing_results) > 0:
    print(f"\n✅ Found {len(existing_results)} existing results!")
    print("   You can use these for analysis or run additional experiments")
    
    # Optionally use existing results instead of running new experiments
    # all_results = existing_results
else:
    print("\n📝 No existing results found. Will run new experiments.")

# Example: Load specific model result
# specific_result = dm.load_individual_result("LLM-GAT/llama-3-8b-instruct-elm-checkpoint-8", "WMDP")
# if specific_result:
#     print(f"Loaded specific result: {specific_result['model_name']} on {specific_result['benchmark']}")


def run_or_load_experiments(force_rerun=False, max_models_per_family=5, subset_size=1000):
    """
    Convenience function to either run new experiments or load existing results.
    
    Args:
        force_rerun: If True, run experiments even if results exist
        max_models_per_family: Maximum models to test per family
        subset_size: Dataset subset size
        
    Returns:
        list: All experiment results
    """
    print(f"🔧 Experiment Manager")
    print(f"   Force rerun: {force_rerun}")
    print(f"   Max models per family: {max_models_per_family}")
    print(f"   Subset size: {subset_size}")
    
    if not force_rerun:
        # Try to load existing results first
        existing_results = dm.load_all_individual_results()
        if len(existing_results) > 0:
            print(f"\n✅ Found {len(existing_results)} existing results!")
            
            # Calculate how many experiments we expect
            expected_total = sum(len(models[:max_models_per_family]) for models in MODEL_CONFIGS.values())
            print(f"   Expected total experiments: {expected_total}")
            print(f"   Found: {len(existing_results)}")
            
            if len(existing_results) >= expected_total:
                print("   📋 All experiments appear to be complete. Using existing results.")
                return existing_results
            else:
                print(f"   ⚠️ Only {len(existing_results)}/{expected_total} experiments found.")
                print("   🚀 Running remaining experiments...")
    
    # Run experiments (new or remaining)
    all_results = run_comprehensive_experiments(
        max_models_per_family=max_models_per_family,
        subset_size=subset_size,
        force_rerun=force_rerun,
        data_manager=dm
    )
    
    return all_results

# Example usage:
# all_results = run_or_load_experiments(force_rerun=False, max_models_per_family=2, subset_size=100)
print("✅ Experiment management function defined")


def display_tables(analysis_results):
    """
    Display formatted tables for analysis results
    
    Args:
        analysis_results: Dictionary containing table dataframes and analysis
    """
    pd.set_option('display.max_rows', None)      # Show all rows
    pd.set_option('display.max_columns', None)   # Show all columns
    pd.set_option('display.width', None)         # Allow unlimited width
    pd.set_option('display.max_colwidth', None)  # Show full column content
    
    print("=" * 80)
    print("📊 TABLE 2: AGGREGATE COMPARISON OF UNLEARNING EVALUATION METRICS")
    print("=" * 80)
    
    table2 = analysis_results['table2']
    print("\n", table2.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("📊 TABLE 3: MODEL FAMILY COMPARISON")
    print("=" * 80)
    
    table3 = analysis_results['table3']
    # Check if table3 is empty before printing
    if not table3.empty:
        print("\n", table3.to_string())
    else:
        print("\n  No data available")
    
    print("\n" + "=" * 80)
    print("📊 TABLE 4: DETAILED INDIVIDUAL RESULTS")
    print("=" * 80)
    
    table4 = analysis_results['table4'].sort_values('AUC', ascending=False)
    # Check if table4 is empty before printing
    if not table4.empty:
        print("\n", table4.to_string(index=False))
    else:
        print("\n  No data available")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    
    stats = analysis_results['summary_stats']
    print(f"Total Evaluations: {stats['total_evaluations']}")
    print(f"Unique Models: {stats['unique_models']}")
    print(f"Unique Benchmarks: {stats['unique_benchmarks']}")
    print(f"Average AUC (LogReg): {stats['avg_auc_logreg']:.3f}")
    print(f"Average AUC (Tree): {stats['avg_auc_tree']:.3f}")
    
    # Performance comparison
    detailed_df = analysis_results['detailed_results']
    
    if not detailed_df.empty:
        print(f"\nBest Performance by Classifier:")
        logreg_data = detailed_df[detailed_df['Classifier']=='LogReg']
        tree_data = detailed_df[detailed_df['Classifier']=='Tree']
        
        if not logreg_data.empty:
            best_logreg = logreg_data['AUC'].max()
            print(f"  LogReg Best AUC: {best_logreg:.3f}")
        
        if not tree_data.empty:
            best_tree = tree_data['AUC'].max()
            print(f"  Tree Best AUC: {best_tree:.3f}")
        
        print(f"\nPerformance by Benchmark:")
        for benchmark in detailed_df['Benchmark'].unique():
            bench_data = detailed_df[detailed_df['Benchmark'] == benchmark]
            if not bench_data.empty:
                avg_auc = bench_data['AUC'].mean()
                print(f"  {benchmark}: {avg_auc:.3f}")
        
        print(f"\nPerformance by Method:")
        for method in detailed_df['Method'].unique():
            method_data = detailed_df[detailed_df['Method'] == method]
            if not method_data.empty:
                avg_auc = method_data['AUC'].mean()
                count = len(method_data)
                print(f"  {method}: {avg_auc:.3f} (n={count})")
    
    print("\n" + "=" * 80)
    
# Debug: Check for problematic values in the data
print("Checking data types and sample values...")
print(f"all_results length: {len(all_results)}")
if len(all_results) > 0:
    print(f"First result keys: {list(all_results[0].keys())}")

# Check for any string 'N/A' values in the first result
for key, value in all_results[0].items():
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str) and v == 'N/A':
                print(f"Found N/A string in {key}.{k}")
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, str) and v2 == 'N/A':
                        print(f"Found N/A string in {key}.{k}.{k2}")
    elif isinstance(value, str) and value == 'N/A':
        print(f"Found N/A string in {key}")

# Let's also check the baselines structure
if 'baselines' in all_results[0]:
    print("Baseline structure:")
    for method, data in all_results[0]['baselines'].items():
        if isinstance(data, dict):
            print(f"  {method}: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, str) and ('N/A' in str(v)):
                    print(f"    Found problematic value: {k}={v}")
        else:
            print(f"  {method}: {data} (not computed)")


def analyze_results_and_generate_tables(all_results):
    """
    Process all_results to generate formatted analysis tables.
    
    Args:
        all_results: List of evaluation results from experiments
        
    Returns:
        dict: Dictionary containing table2, table3, table4, detailed_results, and summary_stats
    """
    
    print(f"\n📊 Processing {len(all_results)} experimental results...\n")
    
    # Generate the comprehensive detailed results
    detailed_df = create_comprehensive_detailed_results(all_results)
    
    # Clean the data: Convert 'N/A' strings and any concatenations (e.g., 'N/AN/A') to np.nan for numeric operations
    numeric_columns = ['Multi_class_AUC', 'Accuracy', 'F1', 'Retain_vs_All_AUC', 'Forget_vs_All_AUC', 'Holdout_vs_All_AUC', 'AUC_at_1_FP', 'Retain_vs_Forget_AUC', 'Retain_vs_Holdout_AUC', 'Forget_vs_Holdout_AUC']
    for col in numeric_columns:
        if col in detailed_df.columns:
            # Replace any string containing 'N/A' (including concatenations like 'N/AN/A') with np.nan
            detailed_df[col] = detailed_df[col].astype(str).replace(r'.*N/A.*', np.nan, regex=True)
            # Convert to numeric, coercing errors to NaN
            detailed_df[col] = pd.to_numeric(detailed_df[col], errors='coerce')
    
    # Rename columns for consistency AFTER cleaning
    column_mapping = {
        'Multi_class_AUC': 'AUC',
        'Evaluation_Method': 'Method',
        'Retain_vs_All_AUC': 'Retain_vs_All',
        'Forget_vs_All_AUC': 'Forget_vs_All',
        'Holdout_vs_All_AUC': 'Holdout_vs_All',
        'Retain_vs_Forget_AUC': 'Retain_vs_Forget',
        'Retain_vs_Holdout_AUC': 'Retain_vs_Holdout',
        'Forget_vs_Holdout_AUC': 'Forget_vs_Holdout'
    }
    detailed_df = detailed_df.rename(columns=column_mapping)
    
    # Calculate baseline averages for Table 2
    baseline_methods = ['Zlib Compression', 'MIN-K%', 'MIN-K%++', 'ROUGE-L F1']
    baseline_avg_data = {}
    
    for method in baseline_methods:
        method_data = detailed_df[detailed_df['Method'] == method]
        if not method_data.empty:
            baseline_avg_data[method] = {
                'Retain_vs_All': method_data['Retain_vs_All'].mean(),
                'Forget_vs_All': method_data['Forget_vs_All'].mean(),
                'Holdout_vs_All': method_data['Holdout_vs_All'].mean(),
                'AUC': method_data['AUC'].mean(),
                'AUC_at_1_FP': method_data['AUC_at_1_FP'].mean()
            }
        else:
            baseline_avg_data[method] = {
                'Retain_vs_All': np.nan,
                'Forget_vs_All': np.nan,
                'Holdout_vs_All': np.nan,
                'AUC': np.nan,
                'AUC_at_1_FP': np.nan
            }
    
    # Calculate REMIND_OURS average results for Table 2
    epp_ue_data = detailed_df[detailed_df['Method'] == 'REMIND_OURS']
    if not epp_ue_data.empty:
        epp_ue_results = epp_ue_data.groupby('Classifier').agg({
            'AUC': 'mean',
            'Accuracy': 'mean', 
            'F1': 'mean',
            'Retain_vs_All': 'mean',
            'Forget_vs_All': 'mean',
            'Holdout_vs_All': 'mean',
            'AUC_at_1_FP': 'mean'
        }).round(3)
    else:
        # Create empty results if no REMIND_OURS data
        epp_ue_results = pd.DataFrame({
            'AUC': [np.nan, np.nan],
            'Accuracy': [np.nan, np.nan],
            'F1': [np.nan, np.nan],
            'Retain_vs_All': [np.nan, np.nan],
            'Forget_vs_All': [np.nan, np.nan],
            'Holdout_vs_All': [np.nan, np.nan],
            'AUC_at_1_FP': [np.nan, np.nan]
        }, index=['LogReg', 'Tree'])
    
    # Table 2: Aggregate Method Comparison
    print("📊 Generating Table 2: Aggregate Method Comparison...\n")
    
    table2_data = {
        'Method': ['Zlib Compression', 'MIN-K%', 'MIN-K%++', 'ROUGE-L F1', 'REMIND_OURS - LogReg (ours)', 'REMIND_OURS - Tree (ours)'],
        'Retain_vs_All_AUC': [
            baseline_avg_data.get('Zlib Compression', {}).get('Retain_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%', {}).get('Retain_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%++', {}).get('Retain_vs_All', np.nan),
            baseline_avg_data.get('ROUGE-L F1', {}).get('Retain_vs_All', np.nan),
            epp_ue_results.loc['LogReg', 'Retain_vs_All'] if 'LogReg' in epp_ue_results.index else np.nan,
            epp_ue_results.loc['Tree', 'Retain_vs_All'] if 'Tree' in epp_ue_results.index else np.nan
        ],
        'Forget_vs_All_AUC': [
            baseline_avg_data.get('Zlib Compression', {}).get('Forget_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%', {}).get('Forget_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%++', {}).get('Forget_vs_All', np.nan),
            baseline_avg_data.get('ROUGE-L F1', {}).get('Forget_vs_All', np.nan),
            epp_ue_results.loc['LogReg', 'Forget_vs_All'] if 'LogReg' in epp_ue_results.index else np.nan,
            epp_ue_results.loc['Tree', 'Forget_vs_All'] if 'Tree' in epp_ue_results.index else np.nan
        ],
        'Holdout_vs_All_AUC': [
            baseline_avg_data.get('Zlib Compression', {}).get('Holdout_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%', {}).get('Holdout_vs_All', np.nan),
            baseline_avg_data.get('MIN-K%++', {}).get('Holdout_vs_All', np.nan),
            baseline_avg_data.get('ROUGE-L F1', {}).get('Holdout_vs_All', np.nan),
            epp_ue_results.loc['LogReg', 'Holdout_vs_All'] if 'LogReg' in epp_ue_results.index else np.nan,
            epp_ue_results.loc['Tree', 'Holdout_vs_All'] if 'Tree' in epp_ue_results.index else np.nan
        ],
        'Multi_class_AUC': [
            baseline_avg_data.get('Zlib Compression', {}).get('AUC', np.nan),
            baseline_avg_data.get('MIN-K%', {}).get('AUC', np.nan),
            baseline_avg_data.get('MIN-K%++', {}).get('AUC', np.nan),
            baseline_avg_data.get('ROUGE-L F1', {}).get('AUC', np.nan),
            epp_ue_results.loc['LogReg', 'AUC'] if 'LogReg' in epp_ue_results.index else np.nan,
            epp_ue_results.loc['Tree', 'AUC'] if 'Tree' in epp_ue_results.index else np.nan
        ],
        'AUC_at_1_FP': [
            baseline_avg_data.get('Zlib Compression', {}).get('AUC_at_1_FP', np.nan),
            baseline_avg_data.get('MIN-K%', {}).get('AUC_at_1_FP', np.nan),
            baseline_avg_data.get('MIN-K%++', {}).get('AUC_at_1_FP', np.nan),
            baseline_avg_data.get('ROUGE-L F1', {}).get('AUC_at_1_FP', np.nan),
            epp_ue_results.loc['LogReg', 'AUC_at_1_FP'] if 'LogReg' in epp_ue_results.index else np.nan,
            epp_ue_results.loc['Tree', 'AUC_at_1_FP'] if 'Tree' in epp_ue_results.index else np.nan
        ]
    }
    table2_df = pd.DataFrame(table2_data)

    # Table 3: Model Family Comparison
    print("📊 Generating Table 3: Model Family Comparison...\n")
    def get_model_family(model_name):
        if 'llama-3' in model_name.lower():
            return 'LLaMA-3-8B'
        elif 'llama-2' in model_name.lower():
            return 'LLaMA-2-7B'
        elif 'zephyr' in model_name.lower():
            return 'Zephyr-7B'
        else:
            return 'Other'

    detailed_df['Model_Family'] = detailed_df['Model'].apply(get_model_family)
    
    # Filter out baseline methods with 'N/A' classifier for family analysis
    family_df = detailed_df[detailed_df['Classifier'] != 'N/A'].copy()
    
    # Ensure all numeric columns are properly converted
    numeric_columns_renamed = ['AUC', 'Accuracy', 'F1', 'Retain_vs_All', 'Forget_vs_All', 'Holdout_vs_All', 'AUC_at_1_FP', 'Retain_vs_Forget', 'Retain_vs_Holdout', 'Forget_vs_Holdout']
    for col in numeric_columns_renamed:
        if col in family_df.columns:
            family_df[col] = pd.to_numeric(family_df[col], errors='coerce')
    
    if not family_df.empty:
        family_results = family_df.groupby(['Model_Family', 'Method', 'Classifier']).agg({
            'AUC': 'mean',
            'Accuracy': 'mean',
            'F1': 'mean',
            'Retain_vs_All': 'mean',
            'Forget_vs_All': 'mean',
            'Holdout_vs_All': 'mean',
            'AUC_at_1_FP': 'mean'
        }).round(3)
    else:
        # Create empty family results if no data
        family_results = pd.DataFrame()

    # Table 4: Detailed Individual Results
    print("📊 Generating Table 4: Detailed Individual Results...\n")
    table4_df = detailed_df.copy()

    # Calculate summary statistics safely
    logreg_data = detailed_df[detailed_df['Classifier'] == 'LogReg']
    tree_data = detailed_df[detailed_df['Classifier'] == 'Tree']
    
    avg_auc_logreg = logreg_data['AUC'].mean() if not logreg_data.empty else np.nan
    avg_auc_tree = tree_data['AUC'].mean() if not tree_data.empty else np.nan

    return {
        'table2': table2_df,
        'table3': family_results,
        'table4': table4_df,
        'detailed_results': detailed_df,
        'summary_stats': {
            'total_evaluations': len(all_results),
            'unique_models': detailed_df['Model'].nunique(),
            'unique_benchmarks': detailed_df['Benchmark'].nunique(),
            'avg_auc_logreg': avg_auc_logreg,
            'avg_auc_tree': avg_auc_tree
        }
    }


# Analyze results and generate tables
if len(all_results) > 0:
    print(f"\n📊 Analyzing {len(all_results)} experimental results...\n")

    # Generate analysis and tables
    analysis_results = analyze_results_and_generate_tables(all_results)
    
    wandb.log({
                "total_evaluations": analysis_results['summary_stats']['total_evaluations'],
                "unique_models": analysis_results['summary_stats']['unique_models'],
                "unique_benchmarks": analysis_results['summary_stats']['unique_benchmarks'],
                "avg_auc_logreg": analysis_results['summary_stats']['avg_auc_logreg'],
                "avg_auc_tree": analysis_results['summary_stats']['avg_auc_tree']
            })
    
    # Log a summary of all_results as a table (e.g., key fields per result)
    results_table = wandb.Table(columns=["model_name", "benchmark", "method", "total_rephrasing_results"])
    for result in all_results:
        results_table.add_data(
            result.get('model_name', 'N/A'),
            result.get('benchmark', 'N/A'),
            result.get('method', 'N/A'),
            len(result.get('rephrasing_results', []))
        )
    wandb.log({"experiment_results_summary": results_table})

    # Display formatted tables
    display_tables(analysis_results)

    # Save results using DataManager
    print(f"\n💾 Saving analysis results using DataManager...\n")
    
    saved_files = dm.save_analysis_results(analysis_results)
    
    # Also save to the legacy format for compatibility
    print(f"\n💾 Also saving to legacy format in results directory...\n")

    # Save each table as a separate CSV and print absolute path
    table2_path = os.path.abspath(f"{RESULTS_DIR}/table2_aggregate_comparison.csv")
    table3_path = os.path.abspath(f"{RESULTS_DIR}/table3_model_family_comparison.csv")
    table4_path = os.path.abspath(f"{RESULTS_DIR}/table4_detailed_results.csv")
    detailed_path = os.path.abspath(f"{RESULTS_DIR}/all_detailed_results.csv")
    json_path = os.path.abspath(f"{RESULTS_DIR}/complete_results.json")

    analysis_results['table2'].to_csv(table2_path, index=False)
    print(f"Table 2 saved to: {table2_path}")

    analysis_results['table3'].to_csv(table3_path)
    print(f"Table 3 saved to: {table3_path}")

    analysis_results['table4'].to_csv(table4_path, index=False)
    print(f"Table 4 saved to: {table4_path}")

    analysis_results['detailed_results'].to_csv(detailed_path, index=False)
    print(f"All detailed results saved to: {detailed_path}")

    # Save complete results as JSON for further analysis
    import json

    def make_json_serializable(obj):
        if isinstance(obj, (dict)):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    with open(json_path, 'w') as f:
        serializable_results = [make_json_serializable(result) for result in all_results]
        json.dump({
            'experiment_results': serializable_results,
            'summary_statistics': make_json_serializable(analysis_results['summary_stats']),
            'configuration': {
                'max_models_per_family': MAX_MODELS_PER_FAMILY,
                'subset_size': SUBSET_SIZE
            }
        }, f, indent=2)
    print(f"Complete results JSON saved to: {json_path}")
    
    print(f"\n📁 Files saved via DataManager:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")

else:
    print("❌ No results to analyze. Check if experiments ran successfully.")





# Create visualizations and final summary
if len(all_results) > 0:
    print(f"\n📈 Creating visualizations...")
    
    # Create performance comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: AUC comparison by model family
    plt.subplot(2, 2, 1)
    detailed_df = analysis_results['detailed_results']
    sns.boxplot(data=detailed_df, x='Model_Family', y='AUC', hue='Classifier')
    plt.title('AUC Performance by Model Family')
    plt.xticks(rotation=45)
    plt.legend(title='Classifier')
    
    # Plot 2: Benchmark comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(data=detailed_df, x='Benchmark', y='AUC', hue='Classifier')
    plt.title('AUC Performance by Benchmark')
    plt.legend(title='Classifier')
    
    # Plot 3: Method comparison
    plt.subplot(2, 2, 3)
    method_means = detailed_df.groupby(['Method', 'Classifier'])['AUC'].mean().reset_index()
    sns.barplot(data=method_means, x='Method', y='AUC', hue='Classifier')
    plt.title('Average AUC by Unlearning Method')
    plt.xticks(rotation=45)
    plt.legend(title='Classifier')
    
    # Plot 4: Overall performance distribution
    plt.subplot(2, 2, 4)
    plt.hist(detailed_df[detailed_df['Classifier']=='LogReg']['AUC'], alpha=0.7, label='LogReg', bins=10)
    plt.hist(detailed_df[detailed_df['Classifier']=='Tree']['AUC'], alpha=0.7, label='Tree', bins=10)
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUC Scores')
    plt.legend()
    
    plt.tight_layout()
    
    # Enhanced saving with DataManager integration
    try:
        # Save main overview plot in multiple formats
        saved_visualizations = {}
        formats = ['png', 'pdf', 'svg']
        
        for fmt in formats:
            plot_path = os.path.join(dm.get_results_dir(), f'performance_overview.{fmt}')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', format=fmt)
            saved_visualizations[f'overview_{fmt}'] = plot_path
            print(f"📊 Overview plot saved as {fmt.upper()}: {plot_path}")
        
        timestamped_path = os.path.join(dm.get_results_dir(), f'performance_overview_{TIMESTAMP}.png')
        plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')
        saved_visualizations['overview_timestamped'] = timestamped_path
        
        plt.show()
        
        # Create additional individual plots using DataManager paths
        print(f"\n📈 Creating individual plots...")
        
        # Individual plot 1: Method performance comparison
        plt.figure(figsize=(12, 6))
        method_data = detailed_df.groupby('Method')['AUC'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(method_data['Method'], method_data['mean'], yerr=method_data['std'], capsize=5)
        plt.title('Method Performance Comparison (Mean ± Std)')
        plt.ylabel('AUC Score')
        plt.xlabel('Unlearning Method')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, method_data['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        method_plot_path = os.path.join(dm.get_results_dir(), 'method_comparison.png')
        plt.savefig(method_plot_path, dpi=300, bbox_inches='tight')
        saved_visualizations['method_comparison'] = method_plot_path
        print(f"📊 Method comparison plot saved: {method_plot_path}")
        plt.show()
        
        # Individual plot 2: Benchmark performance comparison
        plt.figure(figsize=(10, 6))
        benchmark_data = detailed_df.groupby('Benchmark')['AUC'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(benchmark_data['Benchmark'], benchmark_data['mean'], yerr=benchmark_data['std'], capsize=5)
        plt.title('Benchmark Performance Comparison (Mean ± Std)')
        plt.ylabel('AUC Score')
        plt.xlabel('Benchmark')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, benchmark_data['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        benchmark_plot_path = os.path.join(dm.get_results_dir(), 'benchmark_comparison.png')
        plt.savefig(benchmark_plot_path, dpi=300, bbox_inches='tight')
        saved_visualizations['benchmark_comparison'] = benchmark_plot_path
        print(f"📊 Benchmark comparison plot saved: {benchmark_plot_path}")
        plt.show()
        
        # Individual plot 3: Classifier comparison
        plt.figure(figsize=(8, 6))
        classifier_data = detailed_df.groupby('Classifier')['AUC'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(classifier_data['Classifier'], classifier_data['mean'], yerr=classifier_data['std'], capsize=5)
        plt.title('Classifier Performance Comparison (Mean ± Std)')
        plt.ylabel('AUC Score')
        plt.xlabel('Classifier Type')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, classifier_data['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        classifier_plot_path = os.path.join(dm.get_results_dir(), 'classifier_comparison.png')
        plt.savefig(classifier_plot_path, dpi=300, bbox_inches='tight')
        saved_visualizations['classifier_comparison'] = classifier_plot_path
        print(f"📊 Classifier comparison plot saved: {classifier_plot_path}")
        plt.show()
        
        # Save visualization metadata using DataManager
        visualization_metadata = {
            'saved_visualizations': saved_visualizations,
            'creation_timestamp': TIMESTAMP,
            'total_plots': len(saved_visualizations),
            'plot_descriptions': {
                'overview': 'Main 2x2 subplot overview of all metrics',
                'method_comparison': 'Bar chart comparing unlearning methods',
                'benchmark_comparison': 'Bar chart comparing benchmark performance',
                'classifier_comparison': 'Bar chart comparing classifier types'
            }
        }
        
        # Save metadata as JSON
        metadata_path = os.path.join(dm.get_results_dir(), 'visualization_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(visualization_metadata, f, indent=2)
        
        print(f"\n✅ All visualizations saved successfully!")
        print(f"📁 Visualization files saved via DataManager:")
        for viz_type, path in saved_visualizations.items():
            print(f"   {viz_type}: {os.path.basename(path)}")
        print(f"📋 Metadata saved: {os.path.basename(metadata_path)}")
        
    except Exception as e:
        print(f"❌ Error saving visualizations: {e}")
        plt.show()  # Still show the plot even if saving fails
    
    print(f"\n🎯 FINAL SUMMARY")
    print(f"="*60)
    print(f"Comprehensive Unlearning Evaluation Results:")
    print(f"  • Total models evaluated: {analysis_results['summary_stats']['unique_models']}")
    print(f"  • Benchmarks tested: {analysis_results['summary_stats']['unique_benchmarks']}")
    print(f"  • Total evaluations: {analysis_results['summary_stats']['total_evaluations']}")
    print(f"  • Average AUC (LogReg): {analysis_results['summary_stats']['avg_auc_logreg']:.3f}")
    print(f"  • Average AUC (Tree): {analysis_results['summary_stats']['avg_auc_tree']:.3f}")
    
    # Best performing combinations
    best_logreg = detailed_df[detailed_df['Classifier']=='LogReg'].nlargest(1, 'AUC')
    best_tree = detailed_df[detailed_df['Classifier']=='Tree'].nlargest(1, 'AUC')
    
    print(f"\n🏆 BEST PERFORMING COMBINATIONS:")
    print(f"  Logistic Regression:")
    if not best_logreg.empty:
        row = best_logreg.iloc[0]
        print(f"    {row['Model']} on {row['Benchmark']} ({row['Method']}) - AUC: {row['AUC']:.3f}")
    
    print(f"  Random Forest:")
    if not best_tree.empty:
        row = best_tree.iloc[0]
        print(f"    {row['Model']} on {row['Benchmark']} ({row['Method']}) - AUC: {row['AUC']:.3f}")
    
    print(f"\n📊 The tables above replace the dummy values in the original LaTeX tables.")
    print(f"💾 All results and visualizations saved to: {dm.get_results_dir()}/")
    print(f"🎉 Comprehensive evaluation completed successfully!")
    
else:
    print("❌ No results available for visualization and summary.")


# Utility functions for managing visualizations with DataManager
def save_all_visualizations_to_dm(analysis_results, dm, formats=['png', 'pdf']):
    """
    Save all visualizations using DataManager with proper metadata tracking
    
    Args:
        analysis_results: Analysis results from analyze_results_and_generate_tables()
        dm: DataManager instance
        formats: List of formats to save plots in
        
    Returns:
        dict: Dictionary with visualization metadata
    """
    print("💾 Saving all visualizations using DataManager...")
    
    saved_visualizations = {}
    
    # Get data for plotting
    detailed_df = analysis_results['detailed_results']
    
    # 1. Save overview plot (2x2 subplot)
    plt.figure(figsize=(15, 10))
    
    # Plot 1: AUC comparison by model family
    plt.subplot(2, 2, 1)
    sns.boxplot(data=detailed_df, x='Model_Family', y='AUC', hue='Classifier')
    plt.title('AUC Performance by Model Family')
    plt.xticks(rotation=45)
    plt.legend(title='Classifier')
    
    # Plot 2: Benchmark comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(data=detailed_df, x='Benchmark', y='AUC', hue='Classifier')
    plt.title('AUC Performance by Benchmark')
    plt.legend(title='Classifier')
    
    # Plot 3: Method comparison
    plt.subplot(2, 2, 3)
    method_means = detailed_df.groupby(['Method', 'Classifier'])['AUC'].mean().reset_index()
    sns.barplot(data=method_means, x='Method', y='AUC', hue='Classifier')
    plt.title('Average AUC by Unlearning Method')
    plt.xticks(rotation=45)
    plt.legend(title='Classifier')
    
    # Plot 4: Overall performance distribution
    plt.subplot(2, 2, 4)
    plt.hist(detailed_df[detailed_df['Classifier']=='LogReg']['AUC'], alpha=0.7, label='LogReg', bins=10)
    plt.hist(detailed_df[detailed_df['Classifier']=='Tree']['AUC'], alpha=0.7, label='Tree', bins=10)
    plt.xlabel('AUC Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of AUC Scores')
    plt.legend()
    
    plt.tight_layout()
    
    # Save overview in multiple formats
    for fmt in formats:
        overview_path = os.path.join(dm.get_results_dir(), f'performance_overview.{fmt}')
        plt.savefig(overview_path, dpi=300, bbox_inches='tight', format=fmt)
        saved_visualizations[f'overview_{fmt}'] = overview_path
    
    plt.close()  # Close the figure to free memory
    
    # 2. Save method comparison plot
    plt.figure(figsize=(12, 6))
    method_data = detailed_df.groupby('Method')['AUC'].agg(['mean', 'std']).reset_index()
    bars = plt.bar(method_data['Method'], method_data['mean'], yerr=method_data['std'], capsize=5)
    plt.title('Method Performance Comparison (Mean ± Std)')
    plt.ylabel('AUC Score')
    plt.xlabel('Unlearning Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, method_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    method_path = os.path.join(dm.get_results_dir(), 'method_comparison.png')
    plt.savefig(method_path, dpi=300, bbox_inches='tight')
    saved_visualizations['method_comparison'] = method_path
    plt.close()
    
    # 3. Save benchmark comparison plot
    plt.figure(figsize=(10, 6))
    benchmark_data = detailed_df.groupby('Benchmark')['AUC'].agg(['mean', 'std']).reset_index()
    bars = plt.bar(benchmark_data['Benchmark'], benchmark_data['mean'], yerr=benchmark_data['std'], capsize=5)
    plt.title('Benchmark Performance Comparison (Mean ± Std)')
    plt.ylabel('AUC Score')
    plt.xlabel('Benchmark')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, benchmark_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    benchmark_path = os.path.join(dm.get_results_dir(), 'benchmark_comparison.png')
    plt.savefig(benchmark_path, dpi=300, bbox_inches='tight')
    saved_visualizations['benchmark_comparison'] = benchmark_path
    plt.close()
    
    # 4. Save classifier comparison plot
    plt.figure(figsize=(8, 6))
    classifier_data = detailed_df.groupby('Classifier')['AUC'].agg(['mean', 'std']).reset_index()
    bars = plt.bar(classifier_data['Classifier'], classifier_data['mean'], yerr=classifier_data['std'], capsize=5)
    plt.title('Classifier Performance Comparison (Mean ± Std)')
    plt.ylabel('AUC Score')
    plt.xlabel('Classifier Type')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, classifier_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    classifier_path = os.path.join(dm.get_results_dir(), 'classifier_comparison.png')
    plt.savefig(classifier_path, dpi=300, bbox_inches='tight')
    saved_visualizations['classifier_comparison'] = classifier_path
    plt.close()
    
    # Create metadata
    visualization_metadata = {
        'saved_visualizations': saved_visualizations,
        'creation_timestamp': TIMESTAMP,
        'total_plots': len(saved_visualizations),
        'plot_descriptions': {
            'overview': 'Main 2x2 subplot overview of all metrics',
            'method_comparison': 'Bar chart comparing unlearning methods',
            'benchmark_comparison': 'Bar chart comparing benchmark performance',
            'classifier_comparison': 'Bar chart comparing classifier types'
        },
        'data_summary': {
            'total_models': analysis_results['summary_stats']['unique_models'],
            'total_benchmarks': analysis_results['summary_stats']['unique_benchmarks'],
            'total_evaluations': analysis_results['summary_stats']['total_evaluations']
        }
    }
    
    # Save metadata using DataManager
    metadata_path = dm.save_visualizations(visualization_metadata)
    
    print(f"✅ All visualizations saved successfully!")
    print(f"📁 Saved {len(saved_visualizations)} visualization files")
    print(f"📋 Metadata saved to: {os.path.basename(metadata_path)}")
    
    return visualization_metadata

def load_and_display_visualizations(dm, visualization_name=None):
    """
    Load and display visualizations using DataManager
    
    Args:
        dm: DataManager instance
        visualization_name: Specific visualization to load (optional)
        
    Returns:
        dict: Dictionary with available visualizations
    """
    print("🔍 Loading visualizations using DataManager...")
    
    try:
        if visualization_name:
            # Load specific visualization
            matching_files = dm.load_specific_visualization(visualization_name)
            if matching_files:
                print(f"📊 Found {len(matching_files)} files matching '{visualization_name}'")
                return {visualization_name: matching_files}
            else:
                print(f"❌ No visualizations found matching '{visualization_name}'")
                return {}
        else:
            # Load all visualizations
            all_vis = dm.list_visualizations()
            if all_vis:
                print(f"📊 Found visualizations in {len(all_vis)} categories")
                return all_vis
            else:
                print("❌ No visualizations found")
                return {}
                
    except Exception as e:
        print(f"❌ Error loading visualizations: {e}")
        return {}

def display_visualization_summary(dm):
    """
    Display a summary of all saved visualizations
    
    Args:
        dm: DataManager instance
    """
    print("📋 Visualization Summary")
    print("=" * 40)
    
    try:
        # Load metadata
        metadata = dm.load_visualization_metadata()
        
        print(f"📅 Created: {metadata.get('creation_timestamp', 'Unknown')}")
        print(f"📊 Total plots: {metadata.get('total_plots', 0)}")
        
        if 'data_summary' in metadata:
            summary = metadata['data_summary']
            print(f"🔢 Data summary:")
            print(f"   • Models: {summary.get('total_models', 'N/A')}")
            print(f"   • Benchmarks: {summary.get('total_benchmarks', 'N/A')}")
            print(f"   • Evaluations: {summary.get('total_evaluations', 'N/A')}")
        
        if 'plot_descriptions' in metadata:
            print(f"\n📊 Available plots:")
            for plot_type, description in metadata['plot_descriptions'].items():
                print(f"   • {plot_type}: {description}")
        
        # List actual files
        print(f"\n📁 Saved files:")
        if 'saved_visualizations' in metadata:
            for viz_type, path in metadata['saved_visualizations'].items():
                file_size = os.path.getsize(path) / 1024 if os.path.exists(path) else 0
                print(f"   • {viz_type}: {os.path.basename(path)} ({file_size:.1f} KB)")
        
    except FileNotFoundError:
        print("❌ No visualization metadata found")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
    
    print("=" * 40)

print("✅ Visualization utility functions defined!")
print("💡 Use save_all_visualizations_to_dm(), load_and_display_visualizations(), and display_visualization_summary()")


def test_classifier_transferability(
    data_manager,
    sampling_strategy='random',
    n_samples=10,
    fixed_axis=None,
    fixed_value=None,
    test_size=0.2,
    verbose=True
):
    """
    Test transferability of trained classifiers across different models and datasets.
    
    This function loads saved classifiers and tests them on data from different
    model-dataset combinations to evaluate how well classifiers generalize.
    
    Args:
        data_manager: DataManager instance with access to saved classifiers and data
        sampling_strategy: Strategy for selecting pairs to test
            - 'random': Random sampling of train/test pairs (default)
            - 'fixed_model': Fix model, vary dataset (requires fixed_axis='model')
            - 'fixed_dataset': Fix dataset, vary model (requires fixed_axis='dataset')
            - 'cross_only': Only test cross-combinations (different model AND dataset)
            - 'all': Test all possible combinations (WARNING: can be very large)
        n_samples: Number of pairs to sample (used for 'random' strategy)
        fixed_axis: Which axis to fix ('model', 'dataset', or 'neighbor_method')
        fixed_value: Value to fix the axis to (e.g., specific model name)
        test_size: Proportion of data to use for testing
        verbose: Whether to print detailed progress information
        
    Returns:
        pandas.DataFrame: Results table with columns:
            - train_model: Model used for training
            - train_dataset: Dataset used for training  
            - train_neighbor_method: Neighbor method used for training
            - test_model: Model used for testing
            - test_dataset: Dataset used for testing
            - test_neighbor_method: Neighbor method used for testing
            - classifier_type: Type of classifier
            - classifier_name: Name of specific classifier
            - test_accuracy: Accuracy on test data
            - test_auc: AUC on test data
            - is_same_model_type: Whether train/test used same model
            - is_same_dataset: Whether train/test used same dataset
            - is_same_neighbor: Whether train/test used same neighbor method
            
    Examples:
        # Random sampling of 20 pairs
        results = test_classifier_transferability(dm, sampling_strategy='random', n_samples=20)
        
        # Fix model, test on different datasets
        results = test_classifier_transferability(
            dm, 
            sampling_strategy='fixed_model',
            fixed_axis='model',
            fixed_value='llama-3-8b-instruct-elm-checkpoint-8'
        )
        
        # Only test cross-transfer (different model AND dataset)
        results = test_classifier_transferability(dm, sampling_strategy='cross_only')
        
        # Test all combinations (WARNING: can take a long time!)
        results = test_classifier_transferability(dm, sampling_strategy='all')
    """
    
    if verbose:
        print("🔍 Starting classifier transferability analysis...")
        print(f"   Strategy: {sampling_strategy}")
        if sampling_strategy == 'random':
            print(f"   Samples: {n_samples}")
        if fixed_axis:
            print(f"   Fixed axis: {fixed_axis} = {fixed_value}")
    
    # Get all saved classifiers
    classifier_files, classifiers_metadata_dict = data_manager.get_list_classifiers()
    
    if not classifier_files:
        print("❌ No saved classifiers found!")
        return pd.DataFrame()
    
    def filter_data(metadata):
        classifier_type = metadata.get('classifier_type', 'unknown')
        clf_name = metadata.get('clf_name', 'unknown')
        neighbor_method = metadata.get('neighbor_method', 'unknown')
        
        if classifier_type in ['binary_comparisons'] or \
            neighbor_method not in ['token_embedding_proximity']:
            return False
        
        return True
    
    # Convert metadata dict to list format for easier processing
    classifiers_metadata = []
    if classifiers_metadata_dict:
        for filename, metadata in classifiers_metadata_dict.items():
            model_name = metadata.get('model_name', 'unknown')
            if 'llama-3-8b-instruct' in model_name:
                model_type = 'llama-3-8b-instruct-elm-checkpoint-8'
            elif 'Llama-2-7b-chat' in model_name:
                model_type = 'Llama-2-7b-chat'
            elif 'zephyr-7b-beta' in model_name:
                model_type = 'zephyr-7b-beta'
            else:
                model_type = model_name
            
            classifier_type = metadata.get('classifier_type', 'unknown')
            clf_name = metadata.get('clf_name', 'unknown')
            neighbor_method = metadata.get('neighbor_method', 'unknown')
            
            if not filter_data(metadata):
                continue
            classifiers_metadata.append({
                'filename': filename,
                'model': model_type,
                'model_name': model_name,
                'dataset': metadata.get('benchmark_name', 'unknown'),
                'neighbor_method': neighbor_method,
                'classifier_type': classifier_type,
                'clf_name': clf_name
            })
    else:
        # Fallback: parse filenames if metadata not available
        for clf_file in classifier_files:
            parts = clf_file.replace('_classifier.pkl', '').split('_')
            if len(parts) >= 5:
                # Find where dataset starts (TOFU, WMDP, or MUSE)
                dataset_idx = -1
                for i, part in enumerate(parts):
                    if part in ['TOFU', 'WMDP', 'MUSE']:
                        dataset_idx = i
                        break
                
                if dataset_idx == -1:
                    continue
                    
                model = '_'.join(parts[:dataset_idx])
                dataset = parts[dataset_idx]
                neighbor_method = parts[dataset_idx + 1]
                classifier_type = parts[dataset_idx + 2]
                clf_name = '_'.join(parts[dataset_idx + 3:])
                
                if neighbor_method not in ['token_embedding_proximity'] or \
                    clf_name not in ['random_forest']:
                    continue
                
                classifiers_metadata.append({
                    'filename': clf_file,
                    'model': model,
                    'dataset': dataset,
                    'neighbor_method': neighbor_method,
                    'classifier_type': classifier_type,
                    'clf_name': clf_name
                })
                
    # classifiers_metadata = [c for c in classifiers_metadata if c['classifier_type'] in ['random_forest'] and c['neighbor_method'] in ['token_embedding_proximity']]
    
    if verbose:
        print(f"\n📊 Found {len(classifiers_metadata)} classifiers:")
        models = set(c['model'] for c in classifiers_metadata)
        datasets = set(c['dataset'] for c in classifiers_metadata)
        neighbor_methods = set(c['neighbor_method'] for c in classifiers_metadata)
        print(f"   Models: {len(models)}")
        print(f"   Datasets: {len(datasets)}")
        print(f"   Neighbor methods: {len(neighbor_methods)}")
    
    # filter out if 'neighbor_method'!= 'token_embedding_proximity' to reduce combinations
    # classifiers_metadata = [c for c in classifiers_metadata if c['neighbor_method'] == 'token_embedding_proximity']
    # classifiers_metadata = [c for c in classifiers_metadata if c['clf_name'] == 'random_forest']
    
    
    if not classifiers_metadata:
        print("❌ No classifiers found after filtering!")
        raise ValueError("No classifiers found after filtering!")
    
    def filter_pairs(train_clf, test_clf):
        if train_clf["classifier_type"] in ['clustering'] or \
                test_clf["classifier_type"] in ['clustering'] or \
                train_clf["clf_name"] in ['logistic'] or test_clf["clf_name"] in ['logistic'] or \
                train_clf["classifier_type"] != test_clf["classifier_type"] or \
                train_clf == test_clf:
            return False

        return True
    # Generate pairs based on strategy
    pairs = []
    if sampling_strategy == 'all':
        # Test all combinations
        for train_clf in classifiers_metadata:
            for test_clf in classifiers_metadata:
                if train_clf["classifier_type"] == test_clf["classifier_type"]\
                    and train_clf != test_clf:
                    pairs.append((train_clf, test_clf))
        if verbose:
            print(f"\n⚠️  Testing ALL {len(pairs)} combinations (this may take a while!)")
    elif sampling_strategy == 'random':
        # Random sampling
        import random
        
        pairs = []
        seen = set()
        random.shuffle(classifiers_metadata)
        for i, train_clf in enumerate(classifiers_metadata):
            if train_clf["classifier_type"] in ['clustering']:
                continue  # skip clustering classifiers for now
            for test_clf in classifiers_metadata[i+1:]:
                if filter_pairs(train_clf, test_clf):
                    sample_key = tuple([train_clf['filename'], test_clf['filename']])
                    if sample_key in seen:
                        continue  # already sampled
                    seen.add(sample_key)
                    
                    pairs.append((train_clf, test_clf))
                    
                    if len(pairs) == n_samples:
                        break
        
        if verbose:
            print(f"\n🎲 Randomly sampled {len(pairs)} pairs")
            
        pairs = list(pairs)
    
    elif sampling_strategy == 'cross_only':
        # Only test when model AND dataset are different
        for train_clf in classifiers_metadata:
            for test_clf in classifiers_metadata:
                if (train_clf['model'] != test_clf['model'] and 
                    train_clf['dataset'] != test_clf['dataset']):
                    pairs.append((train_clf, test_clf))
        
        if verbose:
            print(f"\n🔀 Testing {len(pairs)} cross-transfer pairs (different model AND dataset)")
    
    elif sampling_strategy in ['fixed_model', 'fixed_dataset', 'fixed_neighbor']:
        # Fix one axis
        if not fixed_axis or not fixed_value:
            raise ValueError("fixed_axis and fixed_value must be provided for fixed strategies")
        
        # Get classifiers matching the fixed value
        fixed_classifiers = [c for c in classifiers_metadata if c[fixed_axis] == fixed_value]
        
        if not fixed_classifiers:
            print(f"❌ No classifiers found with {fixed_axis}={fixed_value}")
            return pd.DataFrame()
        
        # Test on all other combinations
        for train_clf in fixed_classifiers:
            for test_clf in classifiers_metadata:
                pairs.append((train_clf, test_clf))
        
        if verbose:
            print(f"\n📌 Fixed {fixed_axis}={fixed_value}, testing {len(pairs)} pairs")
    
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    # Load all unique test data combinations ONCE before the loop
    # This is much more efficient than loading in the loop
    if verbose:
        print(f"\n📦 Pre-loading test data for all unique model/dataset combinations...")
    
    # Get unique test combinations
    unique_test_combinations = set()
    for _, test_clf in pairs:
        unique_test_combinations.add((test_clf['model_name'], test_clf['dataset']))
    
    # Pre-load all test data
    test_data_cache = {}
    for model, dataset in tqdm(unique_test_combinations, disable=not verbose, desc="Loading test data"):
        try:
            # Reconstruct model name with slashes
            model_name_slashes = model.replace('/', '_')
            
            # Load the saved tensor data
            data_manager.set_subdirectory(f"{model_name_slashes}_{dataset}")
            
            # Load the pre-computed normalized features
            try:
                features_path = os.path.join(data_manager.results_dir, 'post_processed_features_dict.pkl')
                with open(features_path, 'rb') as f:
                    features_dict = pickle.load(f)
                
                # Extract normalized features for each split
                forget_features = features_dict['forget']['normalized_features_tensor']
                retain_features = features_dict['retain']['normalized_features_tensor']
                holdout_features = features_dict['holdout']['normalized_features_tensor']
                
                # Convert to numpy if needed
                if isinstance(forget_features, torch.Tensor):
                    forget_features = forget_features.cpu().numpy()
                if isinstance(retain_features, torch.Tensor):
                    retain_features = retain_features.cpu().numpy()
                if isinstance(holdout_features, torch.Tensor):
                    holdout_features = holdout_features.cpu().numpy()
                    
                # take only test size portion
                min_size = min(len(forget_features), len(retain_features), len(holdout_features))
                test_subset_size = int(min_size * (1 - test_size))
                
                forget_features = forget_features[-test_subset_size:]
                retain_features = retain_features[-test_subset_size:]
                holdout_features = holdout_features[-test_subset_size:]
                
                # Create combined features and labels
                X = np.vstack([forget_features, retain_features, holdout_features])
                y = np.concatenate([
                    np.zeros(len(forget_features)),      # 0 = forget
                    np.ones(len(retain_features)),       # 1 = retain
                    np.full(len(holdout_features), 2)    # 2 = holdout
                ])
                
                test_data_cache[(model, dataset)] = (X, y)
                
                if verbose:
                    print(f"  ✅ Loaded {model}/{dataset}: {X.shape[0]} samples with {X.shape[1]} features")
                    
            except FileNotFoundError:
                if verbose:
                    print(f"  ⚠️  No saved features found for {model}/{dataset}")
                data_manager.return_one_subdir_up()
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Error loading {model}/{dataset}: {e}")
                data_manager.return_one_subdir_up()
            
            data_manager.return_one_subdir_up()
            
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to load {model}/{dataset}: {e}")
    
    if verbose:
        print(f"\n✅ Pre-loaded {len(test_data_cache)} test datasets")
        print(f"\n🧪 Testing {len(pairs)} classifier pairs...")
    
    results = []
    
    
    for train_clf, test_clf in tqdm(pairs, disable=not verbose, desc="Testing pairs"):
        try:
            # Load trained classifier
            clf, clf_metadata = data_manager.load_classifier(train_clf['filename'])
            if clf is None:
                continue
            # if 'random_forest' not in train_clf['clf_name']:
            #     continue  # skip non-random forest classifiers for now
            
            # Get pre-loaded test data
            test_key = (test_clf['model_name'], test_clf['dataset'])
            if test_key not in test_data_cache.keys():
                if verbose:
                    print(f"⚠️  No cached data for {test_clf['model']}/{test_clf['dataset']}")
                continue
            
            X_test, y_test = test_data_cache[test_key]
            
            # 0 = forget, 1 = retain, 2 = holdout
            if clf_metadata['classifier_type'] == 'retain_vs_all':
                y_test = (y_test == 1).astype(int)  # 1 = retain, 0 = forget or holdout
            elif clf_metadata['classifier_type'] == 'forget_vs_all':
                y_test = (y_test == 0).astype(int)  # 1 = forget, 0 = retain or holdout
            elif clf_metadata['classifier_type'] == 'holdout_vs_all':
                y_test = (y_test == 2).astype(int)  # 1 = holdout, 0 = forget or retain
            elif clf_metadata['classifier_type'] == 'overall_predictors':
                # multi-class: keep as is
                pass
            # elif 'retain_vs_all' in clf_metadata['clf_name']:
            #     y_test = (y_test == 1).astype(int)  # 1 = retain, 0 = forget or holdout
            # elif 'forget_vs_all' in clf_metadata['clf_name']:
            #     y_test = (y_test == 0).astype(int)  # 1 = forget, 0 = retain or holdout
            # elif 'holdout_vs_all' in clf_metadata['clf_name']:
            #     y_test = (y_test == 2).astype(int)  # 1 = holdout, 0 = forget or retain
            elif clf_metadata['classifier_type'] == 'retain_vs_forget':
                X_test = X_test[y_test != 2]
                y_test = y_test[y_test != 2]
                y_test = (y_test == 1).astype(int)  # 1 = retain, 0 = forget
            elif clf_metadata['classifier_type'] == 'retain_vs_holdout':
                X_test = X_test[y_test != 0]
                y_test = y_test[y_test != 0]
                y_test = (y_test == 1).astype(int)  # 1 = retain, 0 = holdout
            elif clf_metadata['classifier_type'] == 'forget_vs_holdout':
                X_test = X_test[y_test != 1]
                y_test = y_test[y_test != 1]
                y_test = (y_test == 0).astype(int)  # 1 = forget, 0 = holdout
            else:
                continue
            
            # Skip if no valid data
            if len(X_test) == 0:
                continue
            
            # Make predictions
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate AUC (multi-class)
            auc = None
            if y_pred_proba is not None:
                try:
                    # Check if the number of classes matches
                    n_classes_test = len(np.unique(y_test))
                    n_classes_pred = y_pred_proba.shape[1]
                    
                    if n_classes_test == n_classes_pred:
                        if n_classes_test == 2:
                            # Binary classification: Use positive class probabilities only
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            # Multi-class: Use full probability array with ovr
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    else:
                        if verbose:
                            print(f"  ⚠️  Class mismatch: y_test has {n_classes_test} classes, predictions have {n_classes_pred} classes. Skipping AUC.")
                        auc = None
                        continue
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Could not compute AUC: {e}")
                    auc = None
                            
            # Record results
            results.append({
                'train_model': train_clf['model'],
                'train_model_name': train_clf['model_name'],
                'train_dataset': train_clf['dataset'],
                'train_neighbor_method': train_clf['neighbor_method'],
                'test_model': test_clf['model'],
                'test_model_name': test_clf['model_name'],
                'test_dataset': test_clf['dataset'],
                'test_neighbor_method': test_clf['neighbor_method'],
                'classifier_type': train_clf['classifier_type'],
                'classifier_name': train_clf['clf_name'],
                'test_accuracy': accuracy,
                'test_auc': auc,
                'is_same_model_type': train_clf['model'] == test_clf['model'],
                'is_same_model_name': train_clf['model_name'] == test_clf['model_name'],
                'is_same_dataset': train_clf['dataset'] == test_clf['dataset'],
                'is_same_neighbor': train_clf['neighbor_method'] == test_clf['neighbor_method'],
                'is_same_classifier': train_clf['classifier_type'] == test_clf['classifier_type'] and train_clf['clf_name'] == test_clf['clf_name'],
                'n_test_samples': len(X_test)
            })
            
        except Exception as e:
            if verbose:
                print(f"❌ Error testing {train_clf['filename']} on {test_clf['model']}/{test_clf['dataset']}: {e}")
            continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose and not results_df.empty:
        print(f"\n✅ Completed {len(results_df)} successful tests")
        print(f"\n📊 Summary statistics:")
        print(f"   Average accuracy (same model): {results_df[results_df['is_same_model_type']]['test_accuracy'].mean():.3f}")
        print(f"   Average accuracy (different model): {results_df[~results_df['is_same_model_type']]['test_accuracy'].mean():.3f}")
        print(f"   Average accuracy (same dataset): {results_df[results_df['is_same_dataset']]['test_accuracy'].mean():.3f}")
        print(f"   Average accuracy (different dataset): {results_df[~results_df['is_same_dataset']]['test_accuracy'].mean():.3f}")
    
    return results_df


def analyze_transferability_results(results_df, save_path=None):
    """
    Analyze and visualize transferability test results.
    
    Args:
        results_df: DataFrame from test_classifier_transferability()
        save_path: Optional path to save visualizations
        
    Returns:
        dict: Dictionary containing analysis results and figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if results_df.empty:
        print("❌ No results to analyze")
        return {}
    
    print("📊 Analyzing transferability results...")
    
    analysis = {}
    
    # 1. Overall transfer performance
    print("\n1️⃣ Overall Transfer Performance:")
    print("(results_df['classifier_type'] != 'overall_predictors')")
    cross_unlearning = results_df[
        results_df['is_same_model_type'] & 
        results_df['is_same_dataset'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier'] 
        & (results_df['classifier_type'] != 'overall_predictors')
    ]
    cross_model = results_df[
        ~results_df['is_same_model_type'] & 
        results_df['is_same_dataset'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier'] 
        & (results_df['classifier_type'] != 'overall_predictors')
    ]
    cross_dataset = results_df[
        results_df['is_same_model_type'] & 
        ~results_df['is_same_dataset'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier']
        & (results_df['classifier_type'] != 'overall_predictors')
    ]
    
    full_cross = results_df[
        ~results_df['is_same_model_type'] & 
        ~results_df['is_same_dataset'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier']
        & (results_df['classifier_type'] != 'overall_predictors')
    ]
    
    print(f"   Cross-unlearning (same dataset+neighbor+classifier): {cross_unlearning['test_accuracy'].mean():.3f} ± {cross_unlearning['test_accuracy'].std():.3f} (n={len(cross_unlearning)})")
    print(f"   Cross-model transfer (same dataset): {cross_model['test_accuracy'].mean():.3f} ± {cross_model['test_accuracy'].std():.3f} (n={len(cross_model)})")
    print(f"   Cross-dataset transfer (same model): {cross_dataset['test_accuracy'].mean():.3f} ± {cross_dataset['test_accuracy'].std():.3f} (n={len(cross_dataset)})")
    print(f"   Full cross-transfer (different model & dataset): {full_cross['test_accuracy'].mean():.3f} ± {full_cross['test_accuracy'].std():.3f} (n={len(full_cross)})")
    
    analysis['overall'] = {
        'cross_unlearning': cross_unlearning['test_accuracy'].mean(),
        'cross_model': cross_model['test_accuracy'].mean(),
        'cross_dataset': cross_dataset['test_accuracy'].mean(),
        'full_cross': full_cross['test_accuracy'].mean()
    }
    
    # 2. Transfer by classifier type
    print("\n2️⃣ Transfer Performance by Classifier Type:")
    for clf_type in sorted(results_df['classifier_type'].unique()):
        clf_results = results_df[results_df['classifier_type'] == clf_type]
        print(f"   {clf_type}: {clf_results['test_accuracy'].mean():.3f} ± {clf_results['test_accuracy'].std():.3f}")
    
    # 3. Create visualization for transfer performance
    print("\n3️⃣ Creating transfer performance visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Boxplot by transfer type
    transfer_data = []
    transfer_labels = []
    
    means = []
    stds = []
    
    for name, data in [("Cross-Unlearning", cross_unlearning), 
                       ("Cross-Model", cross_model), 
                       ("Cross-Dataset", cross_dataset),
                       ("Full Cross", full_cross)]:
        if len(data) > 0:
            transfer_data.append(data['test_accuracy'].values)
            transfer_labels.append(name)
            means.append(data['test_accuracy'].mean())
            stds.append(data['test_accuracy'].std())
    
    axes[0].boxplot(transfer_data, labels=transfer_labels)
    axes[0].set_title('Transfer Performance Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Bar plot with error bars
    x_pos = range(len(transfer_labels))
    axes[1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(transfer_labels, rotation=45, ha='right')
    axes[1].set_title('Mean Accuracy by Transfer Type', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[1].text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    analysis['transfer_performance_fig'] = fig
    
    if save_path:
        dm.save_plot(fig, f"{save_path}_transfer_performance")
    
    # 4. Model-to-model transfer matrix
    print("\n4️⃣ Creating model-to-model transfer matrix...")
    
    same_dataset_same_clf = results_df[
        results_df['is_same_dataset'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier']
    ]
    
    if not same_dataset_same_clf.empty:
        transfer_matrix = same_dataset_same_clf.groupby(['train_model', 'test_model'])['test_accuracy'].mean().unstack(fill_value=np.nan)
        
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        sns.heatmap(transfer_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                    vmin=0, vmax=1, ax=ax2, cbar_kws={'label': 'Accuracy'})
        ax2.set_title('Model-to-Model Transfer Performance (Same Dataset)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Test Model', fontsize=12)
        ax2.set_ylabel('Train Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        analysis['model_transfer_matrix'] = transfer_matrix
        analysis['model_transfer_fig'] = fig2
        
        if save_path:
            dm.save_plot(fig2, f"{save_path}_model_transfer")
    
    # 5. Dataset-to-dataset transfer matrix
    print("\n5️⃣ Creating dataset-to-dataset transfer matrix...")
    
    same_model_same_clf = results_df[
        results_df['is_same_model_type'] & 
        results_df['is_same_neighbor'] &
        results_df['is_same_classifier']
    ]
    
    if not same_model_same_clf.empty:
        transfer_matrix_dataset = same_model_same_clf.groupby(['train_dataset', 'test_dataset'])['test_accuracy'].mean().unstack(fill_value=np.nan)
        
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(transfer_matrix_dataset, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Accuracy'})
        ax3.set_title('Dataset-to-Dataset Transfer Performance (Same Model)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Test Dataset', fontsize=12)
        ax3.set_ylabel('Train Dataset', fontsize=12)
        plt.tight_layout()
        
        analysis['dataset_transfer_matrix'] = transfer_matrix_dataset
        analysis['dataset_transfer_fig'] = fig3
        
        if save_path:
            dm.save_plot(fig3, f"{save_path}_dataset_transfer")
    
    print("\n✅ Analysis complete!")
    
    return analysis
def get_available_classifiers_summary(data_manager):
    """
    Get a summary of available classifiers for transferability testing.
    
    Args:
        data_manager: DataManager instance
        
    Returns:
        dict: Summary information about available classifiers
    """
    classifier_files = data_manager.list_classifiers()
    
    if not classifier_files:
        print("❌ No saved classifiers found!")
        return {}
    
    # Parse metadata
    metadata = []
    for clf_file in classifier_files:
        parts = clf_file.replace('_classifier.pkl', '').split('_')
        
        # Find dataset
        dataset_idx = -1
        for i, part in enumerate(parts):
            if part in ['TOFU', 'WMDP', 'MUSE']:
                dataset_idx = i
                break
        
        if dataset_idx == -1:
            continue
        
        model = '_'.join(parts[:dataset_idx])
        dataset = parts[dataset_idx]
        neighbor_method = parts[dataset_idx + 1]
        classifier_type = parts[dataset_idx + 2]
        clf_name = '_'.join(parts[dataset_idx + 3:])
        
        metadata.append({
            'model': model,
            'dataset': dataset,
            'neighbor_method': neighbor_method,
            'classifier_type': classifier_type,
            'clf_name': clf_name
        })
    
    # Create summary
    df = pd.DataFrame(metadata)
    
    print("\n📊 Available Classifiers Summary")
    print("="*80)
    print(f"Total classifiers: {len(df)}")
    print(f"\nModels ({len(df['model'].unique())}):")
    for model in sorted(df['model'].unique()):
        count = len(df[df['model'] == model])
        print(f"   • {model}: {count} classifiers")
    
    print(f"\nDatasets ({len(df['dataset'].unique())}):")
    for dataset in sorted(df['dataset'].unique()):
        count = len(df[df['dataset'] == dataset])
        print(f"   • {dataset}: {count} classifiers")
    
    print(f"\nNeighbor methods ({len(df['neighbor_method'].unique())}):")
    for method in sorted(df['neighbor_method'].unique()):
        count = len(df[df['neighbor_method'] == method])
        print(f"   • {method}: {count} classifiers")
    
    print(f"\nClassifier types ({len(df['classifier_type'].unique())}):")
    for clf_type in sorted(df['classifier_type'].unique()):
        count = len(df[df['classifier_type'] == clf_type])
        print(f"   • {clf_type}: {count} classifiers")
    
    print("="*80)
    
    # Calculate possible combinations
    n_models = len(df['model'].unique())
    n_datasets = len(df['dataset'].unique())
    n_neighbors = len(df['neighbor_method'].unique())
    
    total_possible = len(df) ** 2
    cross_model = len(df) * len(df[df['model'] != df['model'].iloc[0]])
    cross_dataset = len(df) * len(df[df['dataset'] != df['dataset'].iloc[0]])
    
    print(f"\n💡 Transferability Testing Estimates:")
    print(f"   Total possible pairs: {total_possible:,}")
    print(f"   Cross-model pairs: ~{cross_model:,}")
    print(f"   Cross-dataset pairs: ~{cross_dataset:,}")
    print(f"   Recommended random samples: {min(100, total_possible // 10)}")
    print("="*80 + "\n")
    
    return {
        'metadata': df,
        'n_classifiers': len(df),
        'n_models': n_models,
        'n_datasets': n_datasets,
        'n_neighbors': n_neighbors,
        'total_possible_pairs': total_possible
    }


# Updated table5 creation code
import pandas as pd

# Assuming the data is in a list called all_results
# Extract relevant data for REMIND_OURS comparison across neighbor methods
neighbor_methods = []
for result in all_results:
    for rephrasing in result.get('rephrasing_results', []):
        neighbor_method = rephrasing.get('neighbor_method', 'Unknown')
        classifier = rephrasing.get('classifier', 'Unknown')
        
        # Only include logistic for simplicity, or adjust as needed
        if classifier == 'logistic':
            neighbor_methods.append({
                'Neighbor_Method': neighbor_method,
                'Multi_Class_AUC_LogReg': round(rephrasing.get('Multi-class AUC', 0), 3),
                'Retain_vs_All_AUC_LogReg': round(rephrasing.get('Retain vs All AUC', 0), 3),
                'Forget_vs_All_AUC_LogReg': round(rephrasing.get('Forget vs All AUC', 0), 3),
                'Holdout_vs_All_AUC_LogReg': round(rephrasing.get('Holdout vs All AUC', 0), 3),
            })

# Create DataFrame
table_df = pd.DataFrame(neighbor_methods)

if table_df.empty:
    print("No data available for REMIND_OURS neighbor methods comparison.")
else:
    # Group by Neighbor_Method and average the AUCs
    table_df = table_df.groupby('Neighbor_Method').agg({
        'Multi_Class_AUC_LogReg': 'mean',
        'Retain_vs_All_AUC_LogReg': 'mean',
        'Forget_vs_All_AUC_LogReg': 'mean',
        'Holdout_vs_All_AUC_LogReg': 'mean'
    }).round(3).reset_index()

# Display the table
print("Table 5: REMIND_OURS Performance Across Neighbor Generation Methods")
print("=" * 80)
print(table_df.to_string(index=False))

# Optionally save to CSV
table_df.to_csv('table5_neighbor_methods_comparison.csv', index=False)
print("\nTable saved to: table5_neighbor_methods_comparison.csv")


def create_custom_table(all_results):
    """
    Create a custom table with specified columns from all_results.
    
    Args:
        all_results: List of evaluation results from experiments
        
    Returns:
        pandas.DataFrame: Table with the requested columns
    """
    rows = []
    
    for result in all_results:
        model = result['model_name'].split('/')[-1]
        benchmark_name = result['benchmark']
        method_name = result['method']
        
        for rephrasing in result.get('rephrasing_results', []):
            row = {
                'model': model,
                'banchmark_name': benchmark_name,
                'method_name': method_name,
                'neighbor_method': rephrasing.get('neighbor_method', 'Unknown'),
                'classifier': rephrasing.get('classifier', 'Unknown'),
                'Retain vs All AUC': rephrasing.get('Retain vs All AUC', np.nan),
                'Holdout vs All AUC': rephrasing.get('Holdout vs All AUC', np.nan),
                'retain_vs_all_auc_at_1_fp': rephrasing.get('Retain_vs_All_auc_at_1_fp', np.nan),
                'forget_vs_all_auc_at_1_fp': rephrasing.get('Forget_vs_All_auc_at_1_fp', np.nan),
                'holdout_vs_all_auc_at_1_fp': rephrasing.get('Holdout_vs_All_auc_at_1_fp', np.nan)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


rephrasing_options_table = create_custom_table(all_results)


print(rephrasing_options_table)

# ============================================================================
# TRANSFERABILITY TESTING EXAMPLES
# ============================================================================
if TRANSFERABILITY:
    print("\n" + "="*80)
    print("🔬 CLASSIFIER TRANSFERABILITY TESTING")
    print("="*80)

    # First, get a summary of available classifiers
    # print("\n📋 Step 1: Check available classifiers")
    # summary = get_available_classifiers_summary(dm)

    # Example usage of transferability testing
    # Uncomment the examples you want to run:

    # Example 1: Random sampling (recommended for initial exploration)
    # ---------------------------------------------------------------
    print("\n🎲 Example 1: Random sampling of classifier pairs")
    transferability_results = test_classifier_transferability(
        dm,
        sampling_strategy='random',
        n_samples=800,  # Adjust based on summary recommendations
        verbose=True
    )

    if not transferability_results.empty:
        # Save results
        dm.save_to_csv(transferability_results, 'transferability_results')
        
        # Analyze and visualize
        analysis = analyze_transferability_results(
            transferability_results,
            save_path='transferability_random'
        )

    # Example 2: Fix model, test on different datasets
    # ------------------------------------------------
    # print("\n📌 Example 2: Fix model, test on different datasets")
    # # Pick one of your trained models (check summary above)
    # example_model = 'LLM-GAT_llama-3-8b-instruct-elm-checkpoint-8'
    # 
    # transferability_results = test_classifier_transferability(
    #     dm,
    #     sampling_strategy='fixed_model',
    #     fixed_axis='model',
    #     fixed_value=example_model,
    #     verbose=True
    # )
    # 
    # if not transferability_results.empty:
    #     transferability_results.to_csv(f'transferability_{example_model}_fixed.csv', index=False)
    #     analysis = analyze_transferability_results(
    #         transferability_results,
    #         save_path=f'transferability_{example_model}'
    #     )

    # Example 3: Fix dataset, test on different models
    # ------------------------------------------------
    # print("\n📌 Example 3: Fix dataset, test on different models")
    # example_dataset = 'TOFU'
    # 
    # transferability_results = test_classifier_transferability(
    #     dm,
    #     sampling_strategy='fixed_dataset',
    #     fixed_axis='dataset',
    #     fixed_value=example_dataset,
    #     verbose=True
    # )
    # 
    # if not transferability_results.empty:
    #     transferability_results.to_csv(f'transferability_{example_dataset}_fixed.csv', index=False)
    #     analysis = analyze_transferability_results(
    #         transferability_results,
    #         save_path=f'transferability_{example_dataset}'
    #     )

    # Example 4: Only test cross-transfer (different model AND dataset)
    # -----------------------------------------------------------------
    # print("\n🔀 Example 4: Test only cross-transfer (different model AND dataset)")
    # transferability_results = test_classifier_transferability(
    #     dm,
    #     sampling_strategy='cross_only',
    #     verbose=True
    # )
    # 
    # if not transferability_results.empty:
    #     transferability_results.to_csv('transferability_cross_only.csv', index=False)
    #     analysis = analyze_transferability_results(
    #         transferability_results,
    #         save_path='transferability_cross'
    #     )

    # Example 5: Test ALL combinations (WARNING: can take very long!)
    # ---------------------------------------------------------------
    # print("\n⚠️  Example 5: Test ALL combinations (USE WITH CAUTION!)")
    # # Only use this if you have few classifiers or lots of time
    # transferability_results = test_classifier_transferability(
    #     dm,
    #     sampling_strategy='all',
    #     verbose=True
    # )
    # 
    # if not transferability_results.empty:
    #     transferability_results.to_csv('transferability_all.csv', index=False)
    #     analysis = analyze_transferability_results(
    #         transferability_results,
    #         save_path='transferability_all'
    #     )

print("="*80 + "\n")


wandb.finish()

print("finished all tasks!")

