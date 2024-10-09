# data_generation.py

import numpy as np
import random
import torch
import os
import pickle
from sklearn.model_selection import train_test_split

def generate_sequences(transition_matrix, initial_state_probs, num_sequences, seq_length, state_size):
    # Convert transition_matrix and initial_state_probs to tensors for efficient sampling
    transition_matrix = torch.tensor(transition_matrix, dtype=torch.float32)
    initial_state_probs = torch.tensor(initial_state_probs, dtype=torch.float32)

    # Initialize sequences array
    sequences = torch.zeros((num_sequences, seq_length), dtype=torch.long)

    # Sample initial states based on the initial state probabilities
    sequences[:, 0] = torch.multinomial(initial_state_probs, num_samples=num_sequences, replacement=True)

    for t in range(1, seq_length):
        # Get current states
        current_states = sequences[:, t - 1]

        # Get transition probabilities for the current states
        probs = transition_matrix[current_states]  # Shape: [num_sequences, state_size]

        # Sample next states for all sequences at once
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)  # Shape: [num_sequences]

        # Store the next states
        sequences[:, t] = next_states

    return sequences.numpy().tolist()

def prepare_datasets(state_size, seq_lengths, num_sequences, data_dir='data', seed=42, test_size=0.2,return_data = False,leakage=0.2,file_name = 'datasets.pkl'):
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Define non-intersecting subsets for each distribution
    subset_size = state_size // 3
    retain_states = list(range(0, subset_size))
    forget1_states = list(range(subset_size, 2 * subset_size))
    forget2_states = list(range(2 * subset_size, 3 * subset_size))

    # Initialize transition matrices with small leakage
    #leakage is the probability of transitioning to states outside the subset
    retain_transition_matrix = np.full((state_size, state_size), leakage / (state_size - len(retain_states)))
    forget_transition_matrix1 = np.full((state_size, state_size), leakage / (state_size - len(forget1_states)))
    forget_transition_matrix2 = np.full((state_size, state_size), leakage / (state_size - len(forget2_states)))

    # Set higher probabilities for transitions within the subset
    for s in retain_states:
        retain_transition_matrix[s, retain_states] = (1.0 - leakage) / len(retain_states)
    for s in forget1_states:
        forget_transition_matrix1[s, forget1_states] = (1.0 - leakage) / len(forget1_states)
    for s in forget2_states:
        forget_transition_matrix2[s, forget2_states] = (1.0 - leakage) / len(forget2_states)

    # Normalize the transition matrices
    retain_transition_matrix = retain_transition_matrix / retain_transition_matrix.sum(axis=1, keepdims=True)
    forget_transition_matrix1 = forget_transition_matrix1 / forget_transition_matrix1.sum(axis=1, keepdims=True)
    forget_transition_matrix2 = forget_transition_matrix2 / forget_transition_matrix2.sum(axis=1, keepdims=True)

    # Initial state distributions favoring the subsets
    retain_initial_state_probs = np.full(state_size, leakage / (state_size - len(retain_states)))
    retain_initial_state_probs[retain_states] = (1.0 - leakage) / len(retain_states)
    retain_initial_state_probs = retain_initial_state_probs / retain_initial_state_probs.sum()

    forget_initial_state_probs1 = np.full(state_size, leakage / (state_size - len(forget1_states)))
    forget_initial_state_probs1[forget1_states] = (1.0 - leakage) / len(forget1_states)
    forget_initial_state_probs1 = forget_initial_state_probs1 / forget_initial_state_probs1.sum()

    forget_initial_state_probs2 = np.full(state_size, leakage / (state_size - len(forget2_states)))
    forget_initial_state_probs2[forget2_states] = (1.0 - leakage) / len(forget2_states)
    forget_initial_state_probs2 = forget_initial_state_probs2 / forget_initial_state_probs2.sum()

    # Generate sequences
    retain_sequences = generate_sequences(
        retain_transition_matrix, retain_initial_state_probs,
        num_sequences['retain'], seq_lengths['retain'], state_size
    )
    forget_sequences1 = generate_sequences(
        forget_transition_matrix1, forget_initial_state_probs1,
        num_sequences['forget1'], seq_lengths['forget1'], state_size
    )
    forget_sequences2 = generate_sequences(
        forget_transition_matrix2, forget_initial_state_probs2,
        num_sequences['forget2'], seq_lengths['forget2'], state_size
    )

    # Label sequences with their source transition matrix
    retain_sequences_labeled = [('retain', seq) for seq in retain_sequences]
    forget_sequences1_labeled = [('forget1', seq) for seq in forget_sequences1]
    forget_sequences2_labeled = [('forget2', seq) for seq in forget_sequences2]

    # Split retain, forget1, and forget2 sequences into train and test sets
    retain_train_seqs, retain_test_seqs = train_test_split(retain_sequences_labeled, test_size=test_size, random_state=seed)
    forget1_train_seqs, forget1_test_seqs = train_test_split(forget_sequences1_labeled, test_size=test_size, random_state=seed)
    forget2_train_seqs, forget2_test_seqs = train_test_split(forget_sequences2_labeled, test_size=test_size, random_state=seed)

    # Combine forget1 and forget2 sequences
    forget_train_seqs = forget1_train_seqs + forget2_train_seqs
    forget_test_seqs = forget1_test_seqs + forget2_test_seqs

    # Shuffle forget sequences after combining
    random.shuffle(forget_train_seqs)
    random.shuffle(forget_test_seqs)

    # Combine retain and forget sequences
    all_train_sequences = retain_train_seqs + forget_train_seqs
    all_test_sequences = retain_test_seqs + forget_test_seqs

    # Shuffle all combined sequences
    random.shuffle(all_train_sequences)
    random.shuffle(all_test_sequences)

    # Find the maximum sequence length for padding
    max_seq_length = max(seq_lengths.values())

    # Save datasets to files
    data = {
        'retain_train_sequences': retain_train_seqs,
        'retain_test_sequences': retain_test_seqs,
        'forget1_train_sequences': forget1_train_seqs,
        'forget1_test_sequences': forget1_test_seqs,
        'forget2_train_sequences': forget2_train_seqs,
        'forget2_test_sequences': forget2_test_seqs,
        'forget_train_sequences': forget_train_seqs,
        'forget_test_sequences': forget_test_seqs,
        'all_train_sequences': all_train_sequences,
        'all_test_sequences': all_test_sequences,
        'max_seq_length': max_seq_length,
        'state_size': state_size,
        'seq_lengths': seq_lengths,
        'num_sequences': num_sequences,
        'seed': seed,
        'retain_initial_state_probs': retain_initial_state_probs,
        'forget_initial_state_probs1': forget_initial_state_probs1,
        'forget_initial_state_probs2': forget_initial_state_probs2,
        'retain_transition_matrix': retain_transition_matrix,
        'forget_transition_matrix1': forget_transition_matrix1,
        'forget_transition_matrix2': forget_transition_matrix2,
        'leakage': leakage,
    }

    if return_data:
        return data

    


    
    with open(os.path.join(data_dir, file_name), 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Datasets saved to {os.path.join(data_dir, file_name)}")

def load_datasets(data_dir='data',file_name = 'datasets.pkl'):
    # Load datasets from files
    with open(os.path.join(data_dir, file_name), 'rb') as f:
        data = pickle.load(f)
    return data
