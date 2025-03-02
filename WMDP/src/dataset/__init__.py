from collections import defaultdict
from transformers import default_data_collator

from .Base import UnlearnDataset, unlearncollector
from .wmdp import WMDPBio, WMDPCyber, WMDPALL
from .wikitext2 import wikitext
from .gsm8k import GSM8K
from .sst2 import SST2
from .agnews import AGNews

def get_dataset(
    dataset_names,
    tokenizer,
    dataset_seed,
    forget_ratio,
    self_retain=False,
    if_llama=False,
    unlearn_method=None,
):
    ### forget dataset & test dataset
    if dataset_names["forget"] == "wikitext":
        dataset = wikitext("wikitext")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPCyber":
        dataset = WMDPCyber("WMDPCyber", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPBio":
        dataset = WMDPBio("WMDPBio", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]
    elif dataset_names["forget"] == "WMDPALL":
        dataset = GSM8K("GSM8K", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]

    elif dataset_names["forget"] == "GSM8K":
        dataset = GSM8K("GSM8K", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        forget_dataset = dataset["train"]
        test_dataset = dataset["test"]

    elif "forget" not in dataset_names:
        forget_dataset = None
        test_dataset = None
    else:
        raise ValueError("No dataset")

    #### retain dataset
    if dataset_names["retain"] == "wikitext":
        dataset = wikitext("wikitext")
        dataset = dataset.build_dataset(tokenizer)
        retain_dataset = dataset["train"]
    elif dataset_names["retain"] == "WMDPCyber":
        dataset = WMDPCyber("WMDPCyber", subset="retain")
        dataset = dataset.build_dataset(tokenizer)
        retain_dataset = dataset["train"]
    elif dataset_names["retain"] == "WMDPBio":
        dataset = WMDPBio("WMDPBio", subset="retain")
        dataset = dataset.build_dataset(tokenizer)
        retain_dataset = dataset["train"]
    elif dataset_names["retain"] == "WMDPALL":
        dataset = WMDPALL("WMDPALL", subset="retain")
        dataset = dataset.build_dataset(tokenizer)
        retain_dataset = dataset["train"]
    elif dataset_names["retain"] == "GSM8K":
        retain_dataset = None
    elif "retain" not in dataset_names:
        retain_dataset = None
    else:
        raise ValueError("No dataset")

    if "perturb" not in dataset_names or dataset_names["perturb"] is None:
        print("No perturb dataset")
        perturb_dataset = None
    elif dataset_names["perturb"] == "GSM8K":
        dataset = GSM8K("GSM8K", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        perturb_dataset = dataset["train"]
    elif dataset_names["perturb"] == "SST2":
        dataset = SST2("SST2", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        perturb_dataset = dataset["train"]
    elif dataset_names["perturb"] == "AGNews":
        dataset = AGNews("AGNews", subset="forget")
        dataset = dataset.build_dataset(tokenizer)
        perturb_dataset = dataset["train"]
    else:
        raise ValueError("No dataset")   

    unlearn_dataset = UnlearnDataset(
        {"forget": forget_dataset, "retain": retain_dataset, "perturb": perturb_dataset},
        forget_ratio,
        dataset_seed,
        self_retain,
    )
    unlearn_collator = unlearncollector

    test_collator = default_data_collator

    return unlearn_dataset, test_dataset, unlearn_collator, test_collator


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset_names = {"forget": "SafePku", "retain": "BookCorpus"}
    dataset_seed = 8888
    forget_ratio = 0.1
    self_retain = False
    unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
        dataset_names, tokenizer, dataset_seed, forget_ratio, self_retain
    )
    print(len(unlearn_dataset))

    print(len(test_dataset))
    import torch

    dataloader = torch.utils.data.DataLoader(
        unlearn_dataset, batch_size=2, collate_fn=unlearn_collator
    )
    for batch in dataloader:
        print(batch)
        break
