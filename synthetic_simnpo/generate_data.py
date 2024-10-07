# generate_data.py

import argparse
from data_generation import prepare_datasets
import os

def main(args):
    # Configuration parameters
    state_size = args.state_size
    seq_lengths = {
        'retain': args.seq_length_retain,
        'forget1': args.seq_length_forget1,
        'forget2': args.seq_length_forget2,
    }
    num_sequences = {
        'retain': args.num_retain_sequences,
        'forget1': args.num_forget_sequences1,
        'forget2': args.num_forget_sequences2,
    }
    data_dir = args.data_dir
    seed = args.seed
    test_size = args.test_size
    leakage = args.leakage
    




    filename = f"state_size_{state_size}_retain_{seq_lengths['retain']}_{num_sequences['retain']}_forget1_{seq_lengths['forget1']}_{num_sequences['forget1']}_forget2_{seq_lengths['forget2']}_{num_sequences['forget2']}_leakage{leakage}.pkl"
    # Generate and save datasets
    prepare_datasets(state_size, seq_lengths, num_sequences, data_dir, seed, test_size,leakage=leakage,file_name= filename)

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Generate Datasets')

        # Data parameters
        parser.add_argument('--state_size', type=int, default=9, help='Size of the state space (should be a multiple of 3)')
        parser.add_argument('--seq_length_retain', type=int, default=50, help='Sequence length for retain data')
        parser.add_argument('--seq_length_forget1', type=int, default=40, help='Sequence length for forget1 data')
        parser.add_argument('--seq_length_forget2', type=int, default=60, help='Sequence length for forget2 data')
        parser.add_argument('--num_retain_sequences', type=int, default=1000, help='Number of retain sequences')
        parser.add_argument('--num_forget_sequences1', type=int, default=500, help='Number of forget1 sequences')
        parser.add_argument('--num_forget_sequences2', type=int, default=500, help='Number of forget2 sequences')
        parser.add_argument('--data_dir', type=str, default='data', help='Directory to save the datasets')
        parser.add_argument('--leakage', type=float, default=0.2, help='Leakage probability for transitions')

        # Seed parameter
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing')

        args = parser.parse_args()
        main(args)
