#!/bin/bash




# Data parameters
state_size=10
seq_length_retain=20
seq_length_forget1=20
seq_length_forget2=20
num_retain_sequences=10000
num_forget_sequences1=5000
num_forget_sequences2=5000
data_dir="data"
data_seed=42
training_seed=42
unlearning_seed=42
test_size=0.2
leakage=0.2

# Model parameters
n_embd=128
n_layer=4
n_head=4
activation="softmax"



# # Generate data
# echo "Generating datasets..."  ##fewer samples might work

# python generate_data.py \
#     --state_size $state_size \
#     --seq_length_retain $seq_length_retain \
#     --seq_length_forget1 $seq_length_forget1 \
#     --seq_length_forget2 $seq_length_forget2 \
#     --num_retain_sequences $num_retain_sequences \
#     --num_forget_sequences1 $num_forget_sequences1 \
#     --num_forget_sequences2 $num_forget_sequences2 \
#     --data_dir $data_dir \
#     --seed $data_seed \
#     --test_size $test_size \
#     --leakage $leakage









# # # Train the retain/pretrain model
# echo "Training the retain/pretrain model..."

# python train.py \
#     --state_size $state_size \
#     --seq_length_retain $seq_length_retain \
#     --seq_length_forget1 $seq_length_forget1 \
#     --seq_length_forget2 $seq_length_forget2 \
#     --num_retain_sequences $num_retain_sequences \
#     --num_forget_sequences1 $num_forget_sequences1 \
#     --num_forget_sequences2 $num_forget_sequences2 \
#     --data_dir $data_dir \
#     --leakage $leakage \
#     --n_embd $n_embd \
#     --n_layer  $n_layer \
#     --n_head $n_head\
#     --activation $activation \
#     --seed $training_seed \
#     --batch_size 128 \
#     --epochs 5 \
#     --learning_rate 0.0005 \
#     --model_type pretrain \
#     --only_forget1    ##only train on forget1
    


# # # # Evaluate the retain/pretrain/unlearn model
# echo "Evaluating the retain/pretrain/unlearn model..."


# python evaluate.py \
#     --state_size $state_size \
#     --seq_length_retain $seq_length_retain \
#     --seq_length_forget1 $seq_length_forget1 \
#     --seq_length_forget2 $seq_length_forget2 \
#     --num_retain_sequences $num_retain_sequences \
#     --num_forget_sequences1 $num_forget_sequences1 \
#     --num_forget_sequences2 $num_forget_sequences2 \
#     --data_dir $data_dir \
#     --leakage $leakage \
#     --n_embd $n_embd \
#     --n_layer  $n_layer \
#     --n_head $n_head\
#     --activation $activation \
#     --batch_size 128 \
#     --epochs 5 \
#     --learning_rate 0.0005 \
#     --model_type retain \
    
 





# # Unlearn the pretrain model

# echo "Unlearning the pretrain model..."
# python unlearn.py \
#     --state_size $state_size \
#     --seq_length_retain $seq_length_retain \
#     --seq_length_forget1 $seq_length_forget1 \
#     --seq_length_forget2 $seq_length_forget2 \
#     --num_retain_sequences $num_retain_sequences \
#     --num_forget_sequences1 $num_forget_sequences1 \
#     --num_forget_sequences2 $num_forget_sequences2 \
#     --data_dir $data_dir \
#     --leakage $leakage \
#     --n_embd $n_embd \
#     --n_layer  $n_layer \
#     --n_head $n_head\
#     --activation $activation \
#     --pretraining_batch_size 128 \
#     --pretraining_epochs 5 \
#     --pretraining_learning_rate 0.0005 \
#     --loss_type NPO \
#     --seed $unlearning_seed \
#     --unlearning_epochs 1\
#     --batch_size 4 \
#     --learning_rate 0.0005 \
#     --beta 1. \
#     --max_iterations 50 \
#     --use_retrain_eval 










# # # ## plot_unlearning_results

# python plot_evaluation.py \
#     --state_size $state_size \
#     --seq_length_retain $seq_length_retain \
#     --seq_length_forget1 $seq_length_forget1 \
#     --seq_length_forget2 $seq_length_forget2 \
#     --num_retain_sequences $num_retain_sequences \
#     --num_forget_sequences1 $num_forget_sequences1 \
#     --num_forget_sequences2 $num_forget_sequences2 \
#     --data_dir $data_dir \
#     --leakage $leakage \
#     --n_embd $n_embd \
#     --n_layer  $n_layer \
#     --n_head $n_head\
#     --activation $activation \
#     --pretraining_batch_size 128 \
#     --pretraining_epochs 5 \
#     --pretraining_learning_rate 0.0005 \
#     --loss_type NPO \
#     --seed $unlearning_seed \
#     --unlearning_epochs 1\
#     --batch_size 4 \
#     --learning_rate 0.0005 \
#     --beta 1. \
#     --max_iterations 50 \
#     --use_retrain_eval






