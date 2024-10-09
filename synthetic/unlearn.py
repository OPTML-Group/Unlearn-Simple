import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from data_generation import load_datasets
from model import TransformerModel
from evaluate_utils import evaluate_model, evaluate_with_transitions
from utils import OneHotDataset
from unlearn_utils import *
import json
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

    # Use retain and forget test sequences for evaluation
    retain_test_sequences = data['retain_test_sequences']
    forget_test_sequences = data['forget_test_sequences']
    forget_1_test_sequences = data['forget1_test_sequences']
    forget_2_test_sequences = data['forget2_test_sequences']

  

    # Create datasets and dataloaders
    batch_size = args.batch_size

    retain_test_dataset = OneHotDataset(retain_test_sequences, max_seq_length)
    forget_test_dataset = OneHotDataset(forget_test_sequences, max_seq_length)
    forget_1_test_dataset = OneHotDataset(forget_1_test_sequences, max_seq_length)
    forget_2_test_dataset = OneHotDataset(forget_2_test_sequences, max_seq_length)
    

    retain_test_dataloader = DataLoader(retain_test_dataset, batch_size=batch_size, shuffle=False)
    forget_test_dataloader = DataLoader(forget_test_dataset, batch_size=batch_size, shuffle=False)
    forget_1_test_dataloader = DataLoader(forget_1_test_dataset, batch_size=batch_size, shuffle=False)
    forget_2_test_dataloader = DataLoader(forget_2_test_dataset, batch_size=batch_size, shuffle=False)

    # Use forget training sequences for unlearning
    forget_train_sequences = data['forget_train_sequences']
    forget_train_dataset = OneHotDataset(forget_train_sequences, max_seq_length)
    forget_train_dataloader = DataLoader(forget_train_dataset, batch_size=batch_size, shuffle=True)

    # Use retain training sequences for unlearning
    retain_train_sequences = data['retain_train_sequences']
    retain_train_dataset = OneHotDataset(retain_train_sequences, max_seq_length)
    retain_train_dataloader = DataLoader(retain_train_dataset, batch_size=batch_size, shuffle=True)



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

   

    # Load the pretrain model
    directory = f'./models/state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_'
    directory +=   f'{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}'
    

    load_filename = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_lr_{args.pretraining_learning_rate}_bs_{args.pretraining_batch_size}_epoch_{args.pretraining_epochs}_pretrain.pth"
    
    load_model_path = os.path.join(directory, load_filename)
        
    model.load_state_dict(torch.load(load_model_path, map_location=device))
    print(f"Model loaded from {load_model_path}")



    if args.use_retrain_eval:
        ## load retrain model
        retrain_model = TransformerModel(model_config).to(device)
        retrain_filename = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_lr_{args.pretraining_learning_rate}_bs_{args.pretraining_batch_size}_epoch_{args.pretraining_epochs}_retain.pth"
        retrain_model_path = os.path.join(directory, retrain_filename)
        retrain_model.load_state_dict(torch.load(retrain_model_path, map_location=device))
        print(f"Retrain model loaded from {retrain_model_path}")
    else:   
        retrain_model = None


    ## unlearn loss type
    loss_type = args.loss_type  
    
    ## create finetuned model
    if loss_type in ['NPO', 'NPO_KL','NPO_RT']:   
        finetuned_model = TransformerModel(model_config).to(device)
        finetuned_model.load_state_dict(torch.load(load_model_path, map_location=device))
        print(f"Finetuned model loaded from {load_model_path}")

    else:
        finetuned_model = None


    # Set up the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Prepare transition matrices for evaluation
    transition_matrices = {
        'retain': data['retain_transition_matrix'],
        'forget1': data['forget_transition_matrix1'],
        'forget2': data['forget_transition_matrix2'],
    }

    # Evaluation function
    dataloaders = {
        'Retain Test': retain_test_dataloader,
        'Forget Test': forget_test_dataloader,
        'Forget1 Test': forget_1_test_dataloader,
        'Forget2 Test': forget_2_test_dataloader,
    }

    # Evaluate before unlearning
    results_before = evaluate_model(model, dataloaders, criterion, state_size, device)
    kl_divs_before = evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device, retrain_model=retrain_model)

    record_kl = {} ## record kl divergence on test
    with torch.no_grad():
        print(f"Before Unlearning:")
        for name, loss in results_before.items():
            print(f"{name} - Loss: {loss:.4f}")
        for name, kl_div in kl_divs_before.items():
            print(f"{name} - KL Divergence: {kl_div:.4f}")

        for name, kl_div in kl_divs_before.items():
            record_kl[name] = [kl_div]
        record_kl['batch'] = [0]
 





    ## unlearning
   
    total_iterations = 0
    for epoch in range(args.unlearning_epochs):
        print(f"Unlearning Epoch {epoch+1}/{args.unlearning_epochs}")
        total_loss = 0
        model.train()
        count_batches = 0

        for forget_batch, retain_batch in zip(forget_train_dataloader, retain_train_dataloader):


            if total_iterations >= args.max_iterations:
                break




            count_batches += 1
            total_iterations += 1
            optimizer.zero_grad()

            # Prepare forget batch data
            input_ids_f = forget_batch['input_ids'].to(device)
            labels_f = forget_batch['labels'].to(device)
            
            # Prepare retain batch data
            input_ids_r = retain_batch['input_ids'].to(device)
            labels_r = retain_batch['labels'].to(device)

            # Calculate the loss based on the selected loss type
            loss = compute_loss(
                model=model,
                loss_type=loss_type,  # or any other loss type
                X_f = input_ids_f,
                y_f = labels_f,
                X_r = input_ids_r,
                y_r = labels_r,
                finetuned_model=finetuned_model, 
                state_size=state_size,
                beta=args.beta  # For NPO and other losses
            )

            loss.backward()

           

            optimizer.step()
            total_loss += loss.item()

        

            if total_iterations % 1 == 0:
                current_batch_avg_loss = loss.item()/batch_size
                print(f"Current batch average loss: {current_batch_avg_loss:.4f}")

                with torch.no_grad():
                    # Evaluate after each unlearning epoch
                    results_after = evaluate_model(model, dataloaders, criterion, state_size, device)
                    kl_divs_after = evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device,retrain_model=retrain_model)

                    print(f"In Unlearning Epoch {epoch+1}, after batch {count_batches}:")
                    for name, loss in results_after.items():
                        print(f"{name} - Loss: {loss:.4f}")
                    for name, kl_div in kl_divs_after.items():
                        print(f"{name} - KL Divergence: {kl_div:.4f}")

                    for name, kl_div in kl_divs_after.items():
                        record_kl[name].append(kl_div)
                    record_kl['batch'].append(total_iterations)
                    



        if total_iterations >= args.max_iterations:
            break
        
        avg_loss = total_loss / len(forget_train_dataloader)
        print(f"Average unlearning loss: {avg_loss:.4f}")

       



                





























    # Save model state_dict after unlearning
    directory = f'./models/state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_'
    directory += f'{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}'
    

    save_filename = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_lr_{args.pretraining_learning_rate}_"
    save_filename += f'bs_{args.pretraining_batch_size}_epoch_{args.pretraining_epochs}_unlearn.pth'  ## the unlearning model might be overwritten every time after unlearning
                

    torch.save(model.state_dict(), os.path.join(directory, save_filename))
    print(f"Model after unlearning saved to {os.path.join(directory, save_filename)}")
        
    


    ## save kl divergence
    directory1 = f'./record/state_size_{args.state_size}_retain_{args.seq_length_retain}_{args.num_retain_sequences}_forget1_{args.seq_length_forget1}_'
    directory1 +=   f'{args.num_forget_sequences1}_forget2_{args.seq_length_forget2}_{args.num_forget_sequences2}_leakage{args.leakage}/'
    directory2 = f"layer_{args.n_layer}_head_{args.n_head}_embd_{args.n_embd}_{args.activation}_prelr_{args.pretraining_learning_rate}_"
    directory2 += f"prebs_{args.pretraining_batch_size}_preepoch_{args.pretraining_epochs}_unlearn/"
    
    record_filename = f'record_kl_{loss_type}_beta_{args.beta}_iter_{args.max_iterations}_bs_{batch_size}_lr_{args.learning_rate}_seed_{seed}.json'
    os.makedirs(directory1+directory2, exist_ok=True)


    with open(os.path.join(directory1+directory2, record_filename), 'w') as json_file:
        json.dump(record_kl, json_file, indent=4)
        print(f"Test_KL saved to {os.path.join(directory, record_filename)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unlearn Transformer Model')

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

    # Model/pretraining parameters
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='softmax', help='Activation function')
    parser.add_argument('--pretraining_learning_rate', type=float, default=5e-4, help='Pretraining learning rate')
    parser.add_argument('--pretraining_epochs', type=int, default=5, help='Number of pretraining epochs')
    parser.add_argument('--pretraining_batch_size', type=int, default=128, help='Pretraining batch size')


    # Unlearning parameters
    parser.add_argument('--unlearning_epochs', type=int, default=1, help='Number of unlearning epochs')
    parser.add_argument('--max_iterations', type=int, default=100, help='Maximum number of iterations for unlearning')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for unlearning')

    # Seed parameter
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Miscellaneous
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use CUDA even if available')
    parser.add_argument('--loss_type', type=str, default='NPO', help='Type of loss function for unlearning')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta value for NPO and SimNPO and others')
    parser.add_argument('--use_retrain_eval',action='store_true', help='Use retrain model to run evaluation')

    args = parser.parse_args()
    main(args)
