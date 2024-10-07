import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, labels, state_size):
    """Cross-entropy loss with ignore_index=-100."""
    return F.cross_entropy(logits.view(-1, state_size), labels.view(-1), ignore_index=-100) 


def grad_ascent_loss(model, X_f, y_f, state_size):
    """Gradient ascent loss using cross-entropy."""
    logits = model.forward(X_f)  # Forward pass to get logits
    loss = - cross_entropy_loss(logits, y_f, state_size)
    return loss


def grad_descent_loss(model, X_r, y_r, state_size):
    """Gradient descent loss using cross-entropy."""
    logits = model.forward(X_r)  # Forward pass to get logits
    loss = cross_entropy_loss(logits, y_r, state_size)
    return loss


def grad_diff_loss(model, X_f, y_f, X_r, y_r, state_size):
    """Gradient difference loss using cross-entropy."""
    # Forward pass for forget data
    logits_f = model.forward(X_f)
    loss_f = - cross_entropy_loss(logits_f, y_f, state_size)
    

    # Forward pass for retain data
    logits_r = model.forward(X_r)
    loss_r = cross_entropy_loss(logits_r, y_r, state_size)

    return loss_f + loss_r




def grad_diff_kl_forget_loss(model, X_f, y_f, X_r, y_r, finetuned_model, state_size):
    """Gradient difference with KL divergence for forgetting using cross-entropy."""

    logits_f = model.forward(X_f)
    loss_f = -cross_entropy_loss(logits_f, y_f, state_size)  # Cross-entropy with ignore_index=-100


    logits_r = model.forward(X_r)
    loss_r = cross_entropy_loss(logits_r, y_r, state_size)  # Cross-entropy with ignore_index=-100


    # Mask to ignore padding for KL divergence
    valid_mask_f = (y_f != -100).float()  # Mask for valid tokens in forget data

    # KL divergence with fine-tuned model on forget data, ignoring padding
    with torch.no_grad():
        logits_f_finetuned = finetuned_model.forward(X_f)


    kl_forget = F.kl_div(
        F.log_softmax(logits_f, dim=-1) ,  
        F.softmax(logits_f_finetuned, dim=-1) ,
        reduction='none'
    )
    kl_forget = kl_forget.sum(dim=-1) * valid_mask_f  # Apply valid mask to KL divergence
    # Apply mask to logits before computing KL divergence
    kl_forget = kl_forget.sum()/logits_f.size(0)  # Average over the batch

    return loss_f + loss_r +  kl_forget


def kl_loss(model, X_f, y_f, X_r, y_r, finetuned_model, state_size):
    """KL divergence-based retain loss using cross-entropy"""

   
    logits_f = model.forward(X_f)
    loss_f = -cross_entropy_loss(logits_f, y_f, state_size)  # Cross-entropy with ignore_index=-100

    # Mask to ignore padding for KL divergence
    valid_mask_r = (y_r != -100).float()  # Mask for valid tokens in retain data

    # KL divergence with fine-tuned model on retain data, ignoring padding
    logits_r = model.forward(X_r)
    with torch.no_grad():
        logits_r_finetuned = finetuned_model.forward(X_r)

    kl_retain = F.kl_div(
        F.log_softmax(logits_r, dim=-1) ,
        F.softmax(logits_r_finetuned, dim=-1) ,
        reduction='none'
    )
    kl_retain = kl_retain.sum(dim=-1) * valid_mask_r  # Apply valid mask to KL divergence
    kl_retain = kl_retain.sum()/logits_r.size(0)  # Average over the batch

    return loss_f + kl_retain









def NPO_loss(model, X_f, y_f, finetuned_model, beta, state_size):
    """Non-Paired Objective (NPO) loss using cross-entropy and summing over tokens per sample."""

    valid_mask_f = (y_f != -100).float()  # Mask for valid tokens, shape [batch_size, seq_len]

    # Forward pass for forget data with the current model
    logits_f = model.forward(X_f)  # Shape: [batch_size, seq_len, state_size]

    y_f_masked = y_f.clone()
    y_f_masked[y_f_masked == -100] = 0  # Replace -100 with 0 so gather can function

    # Gather the log probabilities corresponding to the true labels
    outputs_f = F.log_softmax(logits_f, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len]

    # Forward pass for forget data with the fine-tuned model
    with torch.no_grad():
        logits_f_finetuned = finetuned_model.forward(X_f)
        outputs_f_finetuned = F.log_softmax(logits_f_finetuned, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)
        assert outputs_f_finetuned.shape  == logits_f_finetuned.shape[:2]  # Check shape

    # Compute the negative log-ratio, applying the valid mask to ignore padding tokens
    neg_log_ratio = (outputs_f_finetuned - outputs_f) * valid_mask_f  # Shape: [batch_size, seq_len]

    # Sum over the tokens per sample (along the sequence length dimension)
    neg_log_ratio_sum = neg_log_ratio.sum(dim=1)  # Shape: [batch_size]
    #import pdb; pdb.set_trace()
    # Compute the NPO loss by averaging over the batch, ignoring padding 
    loss = -F.logsigmoid(beta * neg_log_ratio_sum).mean() * 2 / beta

    return loss




# def NPO_AVE(model, X_f, y_f, finetuned_model, beta, state_size):

#     """Non-Paired Objective (NPO) loss using cross-entropy and average over tokens per sample."""

#     valid_mask_f = (y_f != -100).float()  # Mask for valid tokens, shape [batch_size, seq_len]

#     # Forward pass for forget data with the current model
#     logits_f = model.forward(X_f)  # Shape: [batch_size, seq_len, state_size]

#     y_f_masked = y_f.clone()
#     y_f_masked[y_f_masked == -100] = 0  # Replace -100 with 0 so gather can function

#     # Gather the log probabilities corresponding to the true labels
#     outputs_f = F.log_softmax(logits_f, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len]

#     # Forward pass for forget data with the fine-tuned model
#     with torch.no_grad():
#         logits_f_finetuned = finetuned_model.forward(X_f)
#         outputs_f_finetuned = F.log_softmax(logits_f_finetuned, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)
#         assert outputs_f_finetuned.shape  == logits_f_finetuned.shape[:2]  # Check shape

#     # Compute the negative log-ratio, applying the valid mask to ignore padding tokens
#     neg_log_ratio = (outputs_f_finetuned - outputs_f) * valid_mask_f  # Shape: [batch_size, seq_len]

#     # Average over the tokens per sample (along the sequence length dimension), this is the only line different from NPO_loss
#     neg_log_ratio_sum = neg_log_ratio.sum(dim=1)/valid_mask_f.sum(dim=1)  # Shape: [batch_size]
#     #import pdb; pdb.set_trace()
#     # Compute the NPO loss by averaging over the batch, ignoring padding 
#     loss = -F.logsigmoid(beta * neg_log_ratio_sum).mean() * 2 / beta

#     return loss



# def NPO_NO_REF(model, X_f, y_f, finetuned_model, beta, state_size):

#     """Non-Paired Objective (NPO) loss using cross-entropy and average over tokens per sample."""

#     valid_mask_f = (y_f != -100).float()  # Mask for valid tokens, shape [batch_size, seq_len]

#     # Forward pass for forget data with the current model
#     logits_f = model.forward(X_f)  # Shape: [batch_size, seq_len, state_size]

#     y_f_masked = y_f.clone()
#     y_f_masked[y_f_masked == -100] = 0  # Replace -100 with 0 so gather can function

#     # Gather the log probabilities corresponding to the true labels
#     outputs_f = F.log_softmax(logits_f, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len]

   

#     # Compute the negative log-ratio, applying the valid mask to ignore padding tokens
#     neg_log=  - outputs_f * valid_mask_f  # Shape: [batch_size, seq_len]

#     # Sum over the tokens per sample (along the sequence length dimension) this is the only line different from SimNPO_loss
#     neg_log_sum = neg_log.sum(dim=1)  # Shape: [batch_size]

#     #import pdb; pdb.set_trace()
#     # Compute the NPO loss by averaging over the batch, ignoring padding 
#     loss = -F.logsigmoid(beta * neg_log_sum).mean() * 2 / beta

#     return loss







def NPO_KL(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size):
    """NPO + KL loss."""

   
    forget_loss = NPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)

  
    valid_mask_r = (y_r != -100).float()  # Mask for valid tokens in retain data

    logits_r = model.forward(X_r)  # [batch_size, seq_len, state_size]
    with torch.no_grad():
        logits_r_finetuned = finetuned_model.forward(X_r)


    retain_probs_current = F.log_softmax(logits_r, dim=-1) 
    retain_probs_finetuned = F.softmax(logits_r_finetuned, dim=-1)
    retain_loss = F.kl_div(retain_probs_current, retain_probs_finetuned, reduction='none')

    retain_loss = retain_loss.sum(dim=-1) * valid_mask_r  # Apply mask to KL divergence
    retain_loss = retain_loss.sum()/logits_r.size(0)  # Average over the batch


    # Final NPO + KL loss
    return forget_loss + retain_loss




def NPO_RT(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size):
    """NPO + Retain loss."""


    forget_loss = NPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)

    
    logits_r = model.forward(X_r)  # [batch_size, seq_len, state_size]
    retain_loss = cross_entropy_loss(logits_r, y_r, state_size)


    # Final NPO + retain loss
    return forget_loss + retain_loss




def SimNPO_loss(model, X_f, y_f, finetuned_model, beta, state_size):
    """Non-Paired Objective (NPO) loss using cross-entropy and summing over tokens per sample."""

    valid_mask_f = (y_f != -100).float()  # Mask for valid tokens, shape [batch_size, seq_len]

    # Forward pass for forget data with the current model
    logits_f = model.forward(X_f)  # Shape: [batch_size, seq_len, state_size]

    y_f_masked = y_f.clone()
    y_f_masked[y_f_masked == -100] = 0  # Replace -100 with 0 so gather can function

    # Gather the log probabilities corresponding to the true labels
    outputs_f = F.log_softmax(logits_f, dim=-1).gather(2, y_f_masked.unsqueeze(-1)).squeeze(-1)  # Shape: [batch_size, seq_len]

   

    # Compute the negative log-ratio, applying the valid mask to ignore padding tokens
    neg_log=  - outputs_f * valid_mask_f  # Shape: [batch_size, seq_len]

    # Sum over the tokens per sample (along the sequence length dimension)
    neg_log_sum = neg_log.sum(dim=1)/valid_mask_f.sum(dim=1)  # Shape: [batch_size]

    #import pdb; pdb.set_trace()
    # Compute the NPO loss by averaging over the batch, ignoring padding 
    loss = -F.logsigmoid(beta * neg_log_sum).mean() * 2 / beta

    return loss

def SimNPO_KL(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size):
    """SimNPO + KL loss."""

   
    forget_loss = SimNPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)

  
    valid_mask_r = (y_r != -100).float()  # Mask for valid tokens in retain data

    logits_r = model.forward(X_r)  # [batch_size, seq_len, state_size]
    with torch.no_grad():
        logits_r_finetuned = finetuned_model.forward(X_r)


    retain_probs_current = F.log_softmax(logits_r, dim=-1) 
    retain_probs_finetuned = F.softmax(logits_r_finetuned, dim=-1)
    retain_loss = F.kl_div(retain_probs_current, retain_probs_finetuned, reduction='none')

    retain_loss = retain_loss.sum(dim=-1) * valid_mask_r  # Apply mask to KL divergence
    retain_loss = retain_loss.sum()/logits_r.size(0)  # Average over the batch


    # Final NPO + KL loss
    return forget_loss + retain_loss



def SimNPO_RT(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size):
    """SimNPO + Retain loss."""


    forget_loss = SimNPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)

    
    logits_r = model.forward(X_r)  # [batch_size, seq_len, state_size]
    retain_loss = cross_entropy_loss(logits_r, y_r, state_size)


    # Final NPO + retain loss
    return forget_loss + retain_loss
















def compute_loss(model, loss_type, X_f, y_f, X_r=None, y_r=None, y_idk=None, finetuned_model=None, state_size=None, beta=None):
    """
    Compute loss based on the specified loss type.

    Parameters:
    - model: The current model being trained.
    - loss_type: The type of loss to compute (e.g., 'grad_ascent', 'grad_diff', 'NPO').
    - X_f: Forget data ids.
    - y_f: Forget data labels.
    - X_r: Retain data ids.
    - y_r: Retain data labels.
    - y_idk: IDK data labels.

    - finetuned_model: The pre-trained model for comparison in KL loss functions.
    - state_size: Number of states.
    - beta: Scaling factor used in certain loss functions (e.g., non-paired DPO, kto).

    Returns:
    - Computed loss based on the selected loss_type.
    """

    if loss_type == 'grad_ascent':
        return grad_ascent_loss(model, X_f, y_f, state_size)
    elif loss_type == 'grad_descent':
        return grad_descent_loss(model, X_r, y_r, state_size)
    elif loss_type == 'grad_diff':
        return grad_diff_loss(model, X_f, y_f, X_r, y_r, state_size)
    elif loss_type == 'grad_diff_kl_forget':
        return grad_diff_kl_forget_loss(model, X_f, y_f, X_r, y_r, finetuned_model, state_size)
    elif loss_type == 'kl':
        return kl_loss(model, X_f, y_f, X_r, y_r, finetuned_model, state_size)
    elif loss_type == 'NPO':
        return NPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)
    elif loss_type == 'NPO_KL':
        return NPO_KL(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size)
    elif loss_type == 'NPO_RT':
        return NPO_RT(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size)
    elif loss_type == 'SimNPO':
        return SimNPO_loss(model, X_f, y_f, finetuned_model, beta, state_size)
    elif loss_type == 'SimNPO_KL':
        return SimNPO_KL(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size)
    elif loss_type == 'SimNPO_RT':
        return SimNPO_RT(model, X_f, y_f, X_r, y_r, finetuned_model, beta, state_size)
   
       
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
