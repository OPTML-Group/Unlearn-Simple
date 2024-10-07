import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from data_generation import load_datasets, prepare_datasets
from model import TransformerModel
from evaluate_utils import evaluate_model, evaluate_with_transitions
from utils import OneHotDataset
import os


def main(args):
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device configuration
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    data_dir = args.data_dir
    filename = f"state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}.pkl"
    data = load_datasets(data_dir,file_name= filename)
    print(f"Datasets loaded from {os.path.join(data_dir, filename)}")

    state_size = data['state_size']
    max_seq_length = data['max_seq_length']

   

    if args.model_type == 'retain':
        train_sequences = data['retain_train_sequences']
        test_sequences = data['retain_test_sequences']
    elif args.model_type == 'pretrain':
        if not args.only_forget1:
            train_sequences = data['all_train_sequences']
            test_sequences = data['all_test_sequences']
        else:
            train_sequences = data['forget1_train_sequences'] + data['retain_train_sequences']
            test_sequences = data['forget1_test_sequences'] + data['retain_test_sequences']
            random.shuffle(train_sequences)
            random.shuffle(test_sequences)
       
    else:
        raise ValueError("Invalid model_type. Choose 'retain' or 'pretrain'.")
    


    
   

    # Create datasets and dataloaders
    batch_size = args.batch_size

    train_dataset = OneHotDataset(train_sequences, max_seq_length)
    test_dataset = OneHotDataset(test_sequences, max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   


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

    # Set up the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Prepare transition matrices for evaluation
    transition_matrices = {
        'retain': data['retain_transition_matrix'],
        'forget1': data['forget_transition_matrix1'],
        'forget2': data['forget_transition_matrix2'],
    }

    # Training loop with evaluation after each epoch
    epochs = args.epochs


    ## evaluate before running the model


    with torch.no_grad():
        print(f"Epoch {0} Evaluation:")
        dataloaders = {'Test': test_dataloader}
        results = evaluate_model(model, dataloaders, criterion, state_size, device)
        kl_divs = evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device)

        for name, loss  in results.items():
            print(f"{name} - Loss: {loss:.4f}")
        for name, kl_div in kl_divs.items():
            print(f"{name} - KL Divergence: {kl_div:.4f}")
        
        



    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        total_loss = 0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = criterion(logits.view(-1, state_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_loss:.4f}')


        with torch.no_grad():
            # Evaluate on test set after each epoch
            dataloaders = {'Test': test_dataloader}
            results = evaluate_model(model, dataloaders, criterion, state_size, device)
            kl_divs = evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device)

            print(f"Epoch {epoch+1} Evaluation:")
            for name, loss in results.items():
                print(f"{name} - Loss: {loss:.4f}")
            for name, kl_div in kl_divs.items():
                print(f"{name} - KL Divergence: {kl_div:.4f}")


    # Save model state_dict after training
    directory_a = f'./models/state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_'
    directory_b = f'{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}'
    directory = directory_a + directory_b

    os.makedirs(directory, exist_ok=True)



    save_filename = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_lr_{args.learning_rate}_bs_{args.batch_size}_epoch_{args.epochs}_{args.model_type}.pth"
                
    torch.save(model.state_dict(), os.path.join(directory, save_filename))
    print(f"Model saved to {os.path.join(directory, save_filename)}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer Model')

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








    ## Training type
    parser.add_argument('--model_type', type=str, choices=['retain', 'pretrain'], default='pretrain',
                        help="Type of model to train: 'retain' or 'pretrain'")

    # Model parameters
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function')


    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')

    # Seed parameter
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    # Miscellaneous
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA even if available')
    parser.add_argument('--only_forget1', action='store_true', help='Train only on forget1 data')
    args = parser.parse_args()
    main(args)
