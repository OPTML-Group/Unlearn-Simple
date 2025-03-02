import argparse
import os
import random
import sys
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from fastargs.validation import BoolAsInt, File, Folder, OneOf

sys.path.append("src")

Section("overall", "Overall configs").params(
    model_name=Param(str, required=True, desc="Model name"),
    logger=Param(OneOf(["json", "none"]), default="none", desc="Logger to use"),
    cache_dir=Param(Folder(True), default=".cache", desc="Cache directory"),
    seed=Param(int, default=0, desc="Random seed"),
)

Section("unlearn", "Unlearning configs").params(
    unlearn_method=Param(
        OneOf(
            [
                "FT",
                "Relearn"
            ]
        ),
        default="NPO+FT+SAM",
        desc="Unlearning method",
    ),
    num_epochs=Param(int, default=1, desc="Number of epochs to train"),
    lr=Param(float, default=1e-4, desc="Learning rate"),
    weight_decay=Param(float, default=0.0, desc="Weight decay"),
    gradient_accumulation_steps=Param(
        int, default=1, desc="Gradient accumulation steps"
    ),
    task_name=Param(
        OneOf(["toxic", "copyright", "tofu", "wmdp"]),
        default="toxic",
        desc="Task name",
    ),
    resume_path=Param(
        Folder(False), default=None, desc="Path to resume model for evaluation"
    ),
    max_steps=Param(int, default=-1, desc="Max steps for training"),
    use_lora=Param(BoolAsInt(), default=False, desc="Whether to use LoRA"),
    swa=Param(BoolAsInt(), default=False, desc="Whether to use SWA"),
)

Section("unlearn.NPO+FT", "NPO+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
    beta=Param(float, default=1.0, desc="hyperparameters under NPO loss"),
)

# for sam
Section("unlearn.NPO+FT+SAM", "NPO+FT+SAM unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT+SAM"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
    beta=Param(float, default=1.0, desc="hyperparameters under NPO loss"),
    sam_rho=Param(float, default=0.01, desc="Rho for SAM"),
)

Section("unlearn.NPO+FT+RS", "NPO+FT+RS unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT+RS"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
    beta=Param(float, default=1.0, desc="hyperparameters under NPO loss"),
    sam_rho=Param(float, default=0.01, desc="Rho for SAM"),
)

Section("unlearn.NPO+FT+CR", "NPO+FT+CR unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT+CR"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
    beta=Param(float, default=1.0, desc="hyperparameters under NPO loss"),
    gnr_rho=Param(float, default=0.01, desc="Rho for GNR"),
)

Section("unlearn.NPO+FT+GNR", "NPO+FT+GNR unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT+GNR"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
    beta=Param(float, default=1.0, desc="hyperparameters under NPO loss"),
    gnr_rho=Param(float, default=0.01, desc="Rho for GNR"),
)

Section("unlearn.GA+FT", "GA+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "GA+FT"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before GA loss"),
)

Section("unlearn.GA+FT+SAM", "GA+FT+SAM unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "GA+FT+SAM"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before GA loss"),
    sam_rho=Param(float, default=0.01, desc="Rho for SAM"),
)

Section("dataset", "Dataset configs").params(
    forget_dataset_name=Param(str, default="SafePku", desc="forget dataset name"),
    retain_dataset_name=Param(str, default="TruthfulQA", desc="retain dataset name"),
    perturb_dataset_name=Param(str, default=None, desc="perturb dataset name"),
    dataset_seed=Param(int, default=0, desc="Dataset seed"),
    forget_ratio=Param(float, default=200, desc="Forget ratio"),
    self_retain=Param(BoolAsInt(), default=False, desc="Whether to retain self"),
    batch_size=Param(int, default=16, desc="Batch size"),
)

Section("logger", "General logger configs").params(
    name=Param(
        str,
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        desc="Name of this run",
    ),
)

Section("logger.json", "JSON logger").enable_if(
    lambda cfg: cfg["overall.logger"] == "json"
).params(
    root=Param(Folder(True), default="files/logs", desc="Path to log folder"),
)


class Main:
    def __init__(self) -> None:
        self.make_config()
        self.setup_seed()
        self.init_model()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("LLM unlearning")
        self.config.augment_argparse(parser)
        self.config.collect_argparse_args(parser)

        self.config.validate()
        if not quiet:
            self.config.summary()

    @param("overall.seed")
    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @param("overall.model_name")
    def init_model(self, model_name):
        kwargs = self.config.get_section(f"overall")
        kwargs.update(self.config.get_section(f"unlearn"))
        kwargs.update(self.config.get_section(f"dataset"))
        kwargs.update(self.config.get_section(f"unlearn.{kwargs['unlearn_method']}"))
        kwargs["dataset_names"] = {
            "forget": kwargs["forget_dataset_name"],
            "retain": kwargs["retain_dataset_name"],
            "perturb": kwargs["perturb_dataset_name"],
        }
        self.model = import_module(f"model.relearn").get(**kwargs)

    @param("overall.logger")
    def init_logger(self, logger):
        kwargs = self.config.get_section(f"logger")
        kwargs.update(self.config.get_section(f"logger.{logger}"))
        kwargs["config"] = self.config.get_all_config()
        self.logger = import_module(f"loggers.{logger}_").get(**kwargs)

    def run(self):
        self.model.run(self.logger)


if __name__ == "__main__":
    Main()