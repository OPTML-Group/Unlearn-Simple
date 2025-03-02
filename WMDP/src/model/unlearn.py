import os
import sys
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

sys.path.append("src")
import torch
from peft import  get_peft_model, LoraConfig, AdaLoraConfig, TaskType
from dataset import get_dataset
from metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_toxic,
    eval_wmdp,
)
from unlearn import get_unlearn_method


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class WeightAveragingCallback(TrainerCallback):
    def __init__(self, start_step=100, interval=5):
        self.start_step = start_step
        self.interval = interval
        self.swa_state_dict = None
        self.n = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.start_step and state.global_step % self.interval == 0:
            model = kwargs["model"]
            current_state_dict = {k: p.clone().detach() for k, p in model.state_dict().items()}
            
            if self.swa_state_dict is None:
                self.swa_state_dict = current_state_dict
                self.n = 1
            else:
                for key in self.swa_state_dict:
                    self.swa_state_dict[key] = (self.swa_state_dict[key] * self.n + current_state_dict[key]) / (self.n + 1)
                self.n += 1

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.swa_state_dict is not None:
            model.load_state_dict(self.swa_state_dict, strict=True)


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.gamma = kwargs.get("gamma", None)

        # for sam
        self.beta = kwargs.get("beta", None)
        self.gnr_rho = kwargs.get("gnr_rho", None)
        self.sam_rho= kwargs.get("sam_rho", None)
        self.task_name = kwargs.get("task_name", None)

        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False

        self.swa = kwargs.get("swa", False)

    def init_model(self):
        print(f"Loading the checkpoint from {self.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["q_proj","v_proj"], 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
            self.unlearn_method
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            print("#######################################################")
            print(f"max_steps: {self.max_steps}")
            print("#######################################################")
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            report_to=[],
            full_determinism=True,
            save_total_limit=1,
        )

        if self.swa:
            callbacks = [WeightAveragingCallback(start_step=100, interval=5)]
        else:
            callbacks = None

        self.unlearner = get_unlearn_method(
            name=self.unlearn_method,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.unlearn_dataset,
            eval_dataset=None,
            compute_metrics=None,
            args=training_args,
            data_collator=self.unlearn_collator,
            eval_collector=self.test_collator,
            gamma=self.gamma,
            beta=self.beta,
            if_wanda=self.if_wanda,
            gnr_rho=self.gnr_rho,
            sam_rho=self.sam_rho,
            callbacks=callbacks,
        )


    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            model_name = os.path.join(root, "checkpoints")
 
        eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
        torch.cuda.empty_cache()
        eval_few_shots(model_name=model_name, task_list=["wmdp_bio"],output_path=f"{root}/wmdp.json")

    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            print_trainable_parameters(self.model)
            self.init_dataset()  
            
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)


def get(**kwargs):
    return Unlearn(**kwargs)