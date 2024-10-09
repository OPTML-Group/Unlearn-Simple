import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_generation import load_datasets
from model import TransformerModel
from evaluate_utils import evaluate_model, evaluate_with_transitions
import os

def main(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    data_dir = args.data_dir
    filename = f"state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}.pkl"
    data = load_datasets(data_dir,file_name= filename)
    print(f"Datasets loaded from {os.path.join(data_dir, filename)}")


    state_size = data['state_size']
    max_seq_length = data['max_seq_length']

    # Load test datasets
    retain_test_sequences = data['retain_test_sequences']
    forget_test_sequences = data['forget_test_sequences']
    all_test_sequences = data['all_test_sequences']
    forget_1_test_sequences = data['forget1_test_sequences']
    forget_2_test_sequences = data['forget2_test_sequences']

    # Custom Dataset class with padding
    class OneHotDataset(Dataset):
        def __init__(self, sequences_labeled, max_seq_length):
            self.sequences_labeled = sequences_labeled
            self.max_seq_length = max_seq_length - 1  # Adjust for input_ids and labels length

        def __len__(self):
            return len(self.sequences_labeled)

        def __getitem__(self, idx):
            label, seq = self.sequences_labeled[idx]
            input_ids = seq[:-1]  # All except last token
            labels = seq[1:]      # All except first token

            # Padding
            padding_length = self.max_seq_length - len(input_ids)
            input_ids_padded = input_ids + [0] * padding_length
            labels_padded = labels + [-100] * padding_length  # Use -100 to ignore padding in loss

            return {
                'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
                'labels': torch.tensor(labels_padded, dtype=torch.long),
                'label': label
            }

    # Create datasets and dataloaders
    batch_size = args.batch_size

    retain_test_dataset = OneHotDataset(retain_test_sequences, max_seq_length)
    forget_test_dataset = OneHotDataset(forget_test_sequences, max_seq_length)
    all_test_dataset = OneHotDataset(all_test_sequences, max_seq_length)
    forget_1_test_dataset = OneHotDataset(forget_1_test_sequences, max_seq_length)
    forget_2_test_dataset = OneHotDataset(forget_2_test_sequences, max_seq_length)

    retain_test_dataloader = DataLoader(retain_test_dataset, batch_size=batch_size, shuffle=False)
    forget_test_dataloader = DataLoader(forget_test_dataset, batch_size=batch_size, shuffle=False)
    all_test_dataloader = DataLoader(all_test_dataset, batch_size=batch_size, shuffle=False)
    forget_1_test_dataloader = DataLoader(forget_1_test_dataset, batch_size=batch_size, shuffle=False)
    forget_2_test_dataloader = DataLoader(forget_2_test_dataset, batch_size=batch_size, shuffle=False)

    # Model configuration
    model_config = {
        'state_size': state_size,
        'n_positions': max_seq_length - 1,  # Adjusted for input_ids length
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'activation': args.activation,
    }

    # Initialize the model
    model = TransformerModel(model_config).to(device)



    
    # Load the model
    
    directory = f'./models/state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_'
    directory += f'{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}'
    

    save_filename = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_lr_{args.learning_rate}_bs_{args.batch_size}_epoch_{args.epochs}_{args.model_type}.pth"
    load_model_path = os.path.join(directory, save_filename)

    model.load_state_dict(torch.load(load_model_path, map_location=device))
    print(f"Model loaded from {load_model_path}")



    # Set up the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Prepare transition matrices for evaluation
    transition_matrices = {
        'retain': data['retain_transition_matrix'],
        'forget1': data['forget_transition_matrix1'],
        'forget2': data['forget_transition_matrix2'],
    }

    # Evaluate the model
    dataloaders = {
        'Retain Test': retain_test_dataloader,
        'Forget Test': forget_test_dataloader,
        'All Test': all_test_dataloader,
        'Forget1 Test': forget_1_test_dataloader,
        'Forget2 Test': forget_2_test_dataloader,
    }

    results = evaluate_model(model, dataloaders, criterion, state_size, device)
    kl_divs = evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device)


    print(f"Evaluation Results:")
    for name, loss in results.items():
        print(f"{name} - Loss: {loss:.4f}")
    for name, kl_div in kl_divs.items():
        print(f"{name} - KL Divergence: {kl_div:.4f}")






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Transformer Model')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the datasets')
    parser.add_argument('--state_size', type=int, default=9, help='Size of the state space (should be a multiple of 3)')
    parser.add_argument('--seq_length_retain', type=int, default=50, help='Sequence length for retain data')
    parser.add_argument('--seq_length_forget1', type=int, default=40, help='Sequence length for forget1 data')
    parser.add_argument('--seq_length_forget2', type=int, default=60, help='Sequence length for forget2 data')
    parser.add_argument('--num_retain_sequences', type=int, default=1000, help='Number of retain sequences')
    parser.add_argument('--num_forget_sequences1', type=int, default=500, help='Number of forget1 sequences')
    parser.add_argument('--num_forget_sequences2', type=int, default=500, help='Number of forget2 sequences')
    parser.add_argument('--leakage', type=float, default=0.2, help='Leakage probability for transitions')




    ## Training type
    parser.add_argument('--model_type', type=str, choices=['retain', 'pretrain','unlearn'], default='pretrain',
                        help="Type of model to train: 'retain' or 'pretrain' or 'unlearn'")

    # Model parameters
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function')


    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')

    # Seed parameter
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    # Evaluation parameters
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Evaluation batch size')

    # Miscellaneous

    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA even if available')

    args = parser.parse_args()
    main(args)
