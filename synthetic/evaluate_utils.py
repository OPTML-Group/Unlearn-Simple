# evaluate_utils.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def evaluate_model(model, dataloaders, criterion, state_size, device, retrain_model=None):

    
    model.eval()
    if retrain_model:
        retrain_model.eval()  # Put retrain model in eval mode if provided
    
    results = {}
    with torch.no_grad():
        for name, dataloader in dataloaders.items():

            
            
            total_loss = 0
            total_tokens = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass through the main model
                logits = model(input_ids)  # [batch_size, seq_length, state_size]

               
                if retrain_model:
                    # Forward pass through the retrain model
                    retrain_logits = retrain_model(input_ids)  # [batch_size, seq_length, state_size]
                    
                    # Compute log probabilities of the current model and retrain model
                    log_probs = F.log_softmax(logits, dim=-1)
                    retrain_probs = F.softmax(retrain_logits, dim=-1)

                    # Mask to identify non-padded tokens
                    non_pad_mask = (labels != -100)
                    
                    # Compute KL divergence only on non-padded tokens
                    kl_div = F.kl_div(log_probs, retrain_probs, reduction='none').sum(dim=-1)  # Sum over state_size
                    non_pad_tokens = non_pad_mask.sum().item()

                    total_loss += kl_div[non_pad_mask].sum().item()  # Sum KL divergence for non-padded tokens
                else:
                    # Standard cross-entropy loss for non-padded tokens
                    loss = criterion(logits.view(-1, state_size), labels.view(-1))
                    non_pad_tokens = (labels != -100).sum().item()
                    total_loss += loss.item() * non_pad_tokens

                total_tokens += non_pad_tokens

            avg_loss = total_loss / total_tokens
            results[name] = avg_loss
            
    return results










def evaluate_with_transitions(model, dataloaders, transition_matrices, state_size, device, retrain_model=None):
    
    model.eval()
    if retrain_model:
        retrain_model.eval()  # Set the retrain model in evaluation mode if provided

    results = {}

    with torch.no_grad():
        for name, dataloader in dataloaders.items():

            total_kl_div = 0
            total_tokens = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)  # [batch_size, seq_length]
                labels = batch['labels'].to(device)  # [batch_size, seq_length]
                labels_seq = batch['label']  # List of sequence labels indicating retain, forget1, or forget2

                logits = model(input_ids)  # [batch_size, seq_length, state_size]
                
                
                if retrain_model:
                    # Compute logits for the retrain model
                    retrain_logits = retrain_model(input_ids)  # [batch_size, seq_length, state_size]
                    retrain_probs = torch.softmax(retrain_logits, dim=-1)  # Softmax over logits of the retrain model
                else:
                    # Compute transition probabilities for each sequence from the provided transition matrices
                    batch_size, seq_length = input_ids.shape
                    transition_probs = np.zeros((batch_size, seq_length, state_size))

                    label_to_matrix = {
                        'retain': transition_matrices['retain'],
                        'forget1': transition_matrices['forget1'],
                        'forget2': transition_matrices['forget2']
                    }

                    for i, seq_label in enumerate(labels_seq):
                        trans_matrix = label_to_matrix.get(seq_label)
                        if trans_matrix is None:
                            continue
                        current_states = input_ids[i].cpu().numpy()
                        transition_probs[i] = trans_matrix[current_states]

                    # Convert transition probabilities to torch tensor
                    transition_probs = torch.tensor(transition_probs, dtype=torch.float32, device=device)  # [batch_size, seq_length, state_size]

                # Get the mask for non-padding tokens
                non_pad_mask = (labels != -100)  # [batch_size, seq_length]

                # Compute log probabilities for the main model
                log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, seq_length, state_size]
                
                if retrain_model:
                    # Compute KL divergence between main model and retrain model probabilities
                    kl_div = F.kl_div(log_probs, retrain_probs, reduction='none').sum(-1)  # [batch_size, seq_length]
                else:
                    # Compute KL divergence between main model and transition probabilities
                    kl_div = F.kl_div(log_probs, transition_probs, reduction='none').sum(-1)  # [batch_size, seq_length]

                # Mask out padding tokens
                kl_div = kl_div * non_pad_mask.float()

                total_kl_div += kl_div.sum().item()
                total_tokens += non_pad_mask.sum().item()

            avg_kl_div = total_kl_div / total_tokens   # Average KL divergence over non-padded tokens
            results[name] = avg_kl_div

    return results

