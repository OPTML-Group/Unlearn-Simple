import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("src")
from dataset.dataset import bookcorpus_loaders, get_WikiMIA_dataset
from metrics import eval_few_shots, eval_MIA, eval_ppl, eval_toxic
from trainer import sparsetrainer, trainer


class BaseModel:
    def __init__(self, model_name, cache_dir, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.sparse_training = kwargs["sparse_training"]
        self.recovery = kwargs["recovery"]
        self.lr = kwargs["lr"]
        self.num_warmup_steps = kwargs["num_warmup_steps"]
        self.epochs = kwargs["epochs"]
        self.dataset_name = kwargs["dataset_name"]
        self.batch_size = kwargs["batch_size"]
        self.init_model()

    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_trainer(self):
        if self.sparse_training:
            self.trainer = sparsetrainer(
                self.model, self.lr, self.num_warmup_steps, self.device
            )
        else:
            self.trainer = trainer(
                self.model, self.lr, self.num_warmup_steps, self.device
            )

    def init_loaders(self):
        self.loader = bookcorpus_loaders(self.tokenizer, self.batch_size)

    def prune(self, pruner, logger):
        self.model, sparsity = pruner.prune(self.model, self.tokenizer, self.device)
        results = {"sparsity": sparsity}
        logger.log(results)

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        model_name = os.path.join(root, "checkpoints")
        eval_few_shots(model_name=model_name, output_dir=root)
        eval_ppl(model_name=model_name, output_dir=root)
        eval_toxic(model_name=model_name, output_dir=root)
        MIA_dataset = get_WikiMIA_dataset(32)
        eval_MIA(
            model_name=model_name,
            ref_model_name="facebook/opt-350m",
            dataset=MIA_dataset,
            output_dir=root,
            fraction=0.1,
        )

    def save(self, logger):
        logger.save_ckpt("model", self.model)
        logger.save_ckpt("tokenizer", self.tokenizer)

    def recover(self, logger):
        if self.recovery:
            self.init_trainer()
            self.init_loaders()
            self.model = self.trainer.train(self.loader, self.epochs)
            self.save(logger)


def get(**kwargs):
    return BaseModel(**kwargs)
