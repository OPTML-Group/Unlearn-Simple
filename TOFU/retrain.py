from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainerRetraining
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml, set_random_seed
from datetime import datetime
from finetune import find_all_linear_names, print_trainable_parameters
import pdb

@hydra.main(version_base=None, config_path="config", config_name="retrain")
def main(cfg):
    seed = cfg.seed
    set_random_seed(seed)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True) # save the cfg file
    
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    
    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.retain_type)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")

    if cfg.split == "all":
        steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    else:
        #dont save retain model checkpoint
        steps_per_epoch = max_steps

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//10),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="steps",
            eval_steps = max_steps,
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                                                 torch_dtype=torch.bfloat16, 
                                                 trust_remote_code = True)
    model.generation_config.do_sample = True
    
    if model_cfg["gradient_checkpointing"] == "true": 
        model.gradient_checkpointing_enable()
    
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
    
    trainer = CustomTrainerRetraining(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator,
        eval_cfg=cfg.eval,
        seed=seed,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print(f'Start training: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
    trainer.train()
    trainer.evaluate()
    print(f'Finish training: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

    #save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()

    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()
