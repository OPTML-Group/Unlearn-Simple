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

from torch.utils.data import Subset
import random

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
            print(f"Begin averaging at step {state.global_step}")
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


class Relearn:
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
        self.sam_rho= kwargs.get("sam_rho", None)
        self.gnr_rho = kwargs.get("gnr_rho", None)
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
        if self.max_steps == -1:
            sample_number = 5 * (self.batch_size * self.gradient_accumulation_steps * self.num_devices)
        else:
            # sample_number = 500
            sample_number = self.max_steps * (self.batch_size * self.gradient_accumulation_steps * self.num_devices)
        data_list = random.sample([
                7552, 23729, 22883, 17206, 12062, 2118, 11148, 11920, 23779, 100, 17398, 14311, 1518, 4498, 4887, 4859, 14287, 18307, 20036, 4638, 18886, 946, 15889, 6264, 17896, 21909, 19381, 10699, 11559, 19266, 21990, 19582, 10903, 9370, 22107, 1611, 21968, 22719, 3140, 14264, 8121, 11444, 19336, 10288, 5722, 16623, 13243, 3709, 10289, 1891, 23881, 14821, 13117, 13682, 3473, 15105, 14995, 23230, 7856, 22178, 6160, 15185, 23299, 5890, 3790, 12731, 18574, 16694, 9234, 23940, 13898, 22430, 3013, 23983, 9190, 19135, 1093, 7024, 5165, 6558, 16028, 21211, 1022, 4865, 15467, 5554, 1468, 11502, 11607, 22314, 13057, 16371, 22832, 2667, 10405, 160, 934, 20614, 16258, 21690, 14803, 2451, 3094, 26, 6176, 19861, 17526, 18868, 11194, 20551, 18277, 10790, 8179, 16277, 15669, 19731, 24199, 13446, 13573, 5568, 1260, 12643, 926, 12010, 19440, 24236, 11987, 1021, 9113, 15977, 22823, 20425, 16851, 1758, 19754, 1707, 22777, 21469, 22766, 3937, 10088, 5239, 13950, 18320, 16130, 6148, 15679, 18860, 10018, 1844, 15336, 23634, 5195, 20584, 2075, 20836, 9956, 5809, 15047, 19943, 21100, 20397, 7979, 23472, 13291, 6632, 12443, 21323, 22240, 12042, 21513, 9466, 4431, 17007, 8898, 2384, 21051, 6691, 22097, 3288, 1214, 16342, 23255, 17320, 24040, 5646, 10440, 10779, 16824, 10393, 19028, 21630, 4421, 21632, 19703, 13972, 10888, 3478, 9276, 14535, 13613, 168, 20750, 12565, 999, 23022, 18370, 16192, 24336, 3938, 24290, 14351, 4413, 10747, 18851, 7095, 18219, 5956, 15199, 16765, 23608, 3955, 22253, 15019, 23633, 536, 13312, 5360, 16811, 17383, 21413, 3264, 8336, 11480, 1365, 21964, 5449, 18893, 7245, 2650, 14519, 20083, 936, 2923, 17636, 23427, 9500, 7214, 6496, 21763, 9678, 1025, 4477, 10039, 19186, 11064, 2015, 3524, 8180, 13086, 4607, 14641, 8840, 13717, 21026, 20367, 11569, 12874, 558, 17330, 6670, 8720, 16270, 24292, 6510, 23641, 736, 8057, 15813, 8424, 24084, 1825, 3952, 1695, 18636, 4146, 24049, 14474, 1408, 14966, 20677, 16357, 23418, 23727, 20127, 9357, 13824, 14915, 20720, 23526, 21374, 10448, 14309, 6705, 1524, 9225, 22960, 3699, 10146, 1497, 16894, 5329, 4633, 9508, 12666, 53, 20925, 8480, 2668, 17251, 5524, 12602, 4258, 17443, 831, 5005, 22556, 365, 5988, 12814, 23982, 23655, 12171, 7713, 11938, 1861, 277, 17195, 13122, 7712, 2250, 4808, 22705, 7606, 5582, 13601, 8117, 923, 21618, 17422, 10544, 176, 2290, 14421, 10459, 5665, 16449, 6413, 14486, 2937, 914, 16738, 19727, 19233, 4937, 2261, 14557, 7173, 2961, 5367, 16099, 21717, 17424, 4921, 17640, 2566, 6003, 5342, 17829, 15270, 6863, 3594, 1398, 17078, 5805, 18720, 14725, 5925, 24319, 12653, 23124, 21417, 12759, 16143, 15590, 21203, 11111, 17940, 14956, 15850, 20247, 22276, 1921, 15036, 12752, 19684, 21242, 17504, 24302, 3012, 18613, 167, 21972, 23438, 6430, 17882, 1792, 12243, 24188, 3122, 10867, 13278, 24352, 12685, 8197, 6224, 8596, 22125, 24422, 13853, 18459, 15282, 19531, 14105, 11994, 3039, 21953, 1633, 12127, 2200, 19314, 17379, 4518, 12526, 19571, 5514, 1034, 7143, 6854, 18000, 8203, 20138, 17703, 4124, 13937, 22977, 16456, 19380, 22477, 23251, 18570, 18024, 15378, 13593, 16607, 8973, 23671, 15401, 1031, 14372, 10239, 23371, 213, 17788, 22204, 14686, 3922, 15082, 14867, 4851, 2195, 17431, 84, 14582, 23933, 701, 8295, 10848, 8326, 5152, 12737, 21448, 519, 1234, 8516, 1118, 3246, 9099, 20758, 23856
                ], sample_number)
        self.unlearn_dataset = Subset(unlearn_dataset, data_list)

        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(self.unlearn_dataset)) // (
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
        eval_few_shots(model_name=model_name, task_list=["wmdp"],output_path=f"{root}/wmdp.json")

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
    return Relearn(**kwargs)