# SimNPO

## TOFU
### Installation
```
cd TOFU

conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Finetune your models
The code currently supports `Phi-1.5`, and `Llama2-7b chat` models. But newer models can directly be added in the `model_config.yaml` file. For the unlearning challenege, we fine-tuned `Phi-1.5` for 5 epochs using a maximum learning rate of `2e-5`, and the `Llama2-7b chat` model for the same duration at `1e-5`. Finetuning can be done as follows:

```
master_port=18765
split=full
model=llama2-7b
lr=1e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=1 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

### Unlearned your models
```
python run_scripts.py
```


### Get yours results
```
python get_tofu_results.py
```

The results are stored in `TOFU/aggregated_metrics_pandas_62.csv` and `TOFU/aggregated_metrics_pandas_125.csv`.



<!-- # BLUE
```bash
conda activate tofu
cd TOFU

# ft
model=phi
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 finetune.py --config-name=finetune.yaml split=full batch_size=1 gradient_accumulation_steps=4 model_family=llama2-7b

# npo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 evaluate_util.py model_path=results/locuslab/tofu_ft_llama2-7b/8GPU_grad_diff_1e-05_forget10_epoch10_batch1_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 evaluate_util.py model_path=results/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/8GPU_npo_grad_diff_1e-05_forget01_epoch10_batch4_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1

# grad_diff
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=grad_diff

# idk
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=idk

# npo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 vector_forget.py --config-name=vector_forget.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=18765 vector_forget.py --config-name=vector_forget.yaml

# bash
bash /egr/research-optml/chongyu/NEW-BLUE/TOFU/commands/run0.sb&

# debug
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget.py --config-name=forget.yaml forget_loss=grad_diff save_dir=try

# generate mask
delete `ddp_find_unused_parameters= False, deepspeed='config/ds_config.json',` in vector_forget.py
delete `self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)` in dataloader.py
delete `def create_optimizer(self):` in dataloader.py

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 vector_forget.py --config-name=vector_forget.yaml lower_level_times=0

# plot loss
## tofu
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=18765 plot_loss_dist.py --config-name=forget.yaml model_path=/egr/research-optml/chongyu/NEW-BLUE/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/8GPU_simnpo_grad_diff_1e-05_forget05_epoch10_batch1_accum4_beta5.0_grad_diff_coeff0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1/checkpoint-62/ eval.batch_size=1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=18765 plot_loss_dist.py --config-name=forget.yaml model_path=/egr/research-optml/chongyu/NEW-BLUE/TOFU/paper_models/final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-625/8GPU_npo_grad_diff_1e-05_forget05_epoch20_batch1_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1/checkpoint-125/ eval.batch_size=1
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=18765 plot_loss_dist.py --config-name=forget.yaml eval.batch_size=1

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=18765 plot_loss_dist.py --config-name=forget.yaml model_path=/egr/research-optml/chongyu/NEW-BLUE/TOFU/retrained_models/forget05_final_ft_noLORA_5_epochs_inst_lr1e-05_llama2-7b_full_seed42_1/checkpoint-148 eval.batch_size=1

## wmdp
CUDA_VISIBLE_DEVICES=0 python plot_loss_dist.py --algo simnpo --model_dir muse-bench/MUSE-Books_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/books/raw/forget.txt --out_dir "./json/books/simnpo_forget.json" --max_len 2048


# retain
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 retrain.py --config-name=retrain.yaml

# wmdp
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18765 forget_wmdp.py --config-name=forget_wmdp.yaml

# wmdp evaluation
python lm_eval --model hf --model_args pretrained=/egr/research-optml/wangc168/watermark/wmdp/models/new_rmu_natural   --tasks wmdp_bio --device cuda:6 --batch_size 16

python lm_eval --model hf --model_args pretrained=/egr/research-optml/chongyu/NEW-BLUE/TOFU/wmdp_models/origin/unlearned/8GPU_npo_grad_diff_1e-06_wmdp_epoch10_batch1_accum4_beta0.1_grad_diff_coeff1.0_reffine_tuned_evalsteps_per_epoch_seed1001_1 --tasks wmdp_bio --device cuda:6 --batch_size 16

torchrun --nproc-per-node=8 --no-python lm_eval \
    --model hf \
    --model_args pretrained=/egr/research-optml/chongyu/NEW-BLUE/TOFU/wmdp_models/origin/unlearned/8GPU_simnpo_grad_diff_1e-06_wmdp_epoch10_batch1_accum4_beta10.0_grad_diff_coeff0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1/checkpoint-500 \
    --output_path try \
    --tasks wmdp_bio \
    --batch_size auto
``` -->