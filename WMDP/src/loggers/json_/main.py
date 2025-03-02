import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM

from ..base import BaseLogger


class JSONLogger(BaseLogger):
    def __init__(self, root, name, config):
        root = os.path.join(root, name)
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.ckpt_root = os.path.join(root, "checkpoints")
        self.img_root = os.path.join(root, "images")
        os.makedirs(self.ckpt_root, exist_ok=True)
        os.makedirs(self.img_root, exist_ok=True)
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            f.flush()

        self.log_path = os.path.join(root, "log.json")
        self.start_time = datetime.now()
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data_last = json.load(f)[-1]
            self.start_time -= datetime.strptime(
                old_data_last["current_time"], "%Y-%m-%d-%H-%M-%S"
            ) - datetime.strptime(old_data_last["start_time"], "%Y-%m-%d-%H-%M-%S")

    def log(self, data):
        cur_time = datetime.now()
        stats = {
            "start_time": self.start_time.strftime("%Y-%m-%d-%H-%M-%S"),
            "current_time": cur_time.strftime("%Y-%m-%d-%H-%M-%S"),
            "relative_time": str(cur_time - self.start_time),
            **data,
        }
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data = json.load(f)
            with open(self.log_path, "w") as f:
                json.dump(old_data + [stats], f, indent=4)
                f.flush()
        else:
            with open(self.log_path, "w") as f:
                json.dump([stats], f, indent=4)
                f.flush()
        print("logging:", stats)

    def truncate(self, epoch):
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data = json.load(f)
            with open(self.log_path, "w") as f:
                json.dump(old_data[:epoch], f, indent=4)
                f.flush()
        else:
            assert epoch == 0

    def save_ckpt(self, name, model, use_lora):
        if "tokenizer" not in name and use_lora:
            model = model.merge_and_unload()
        model.save_pretrained(self.ckpt_root)

    def load_ckpt(self, name, device="cpu"):
        model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_root,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        return model

    def clear_ckpt_root(self):
        from shutil import rmtree

        rmtree(self.ckpt_root)
        os.makedirs(self.ckpt_root, exist_ok=True)

    def save_img(self, name, img):
        path = os.path.join(self.img_root, f"{name}.png")
        img.save(path)

    def get_root(self):
        return os.path.abspath(self.root)


def test():
    logger = JSONLogger("./", "test")
    logger.log({"a": 1})
    logger.log({"b": 2})
    logger.log({"c": 3})


def get(**kwargs):
    return JSONLogger(**kwargs)
