# MUSE

## Installation

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate muse
```

## Get the data & origin models

Two corpora `News` and `Books` and the associated target models are available as follows:

| Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
|----------|:------------------------------:|----------| 
| News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

Before proceeding, load all the data from HuggingFace to the root of this repostiory by running the following instruction:
```
python load_data.py
```

## Get the unlearned model

Run `unlearn.py` in the `baselines` folder.
```python
# news
python unlearn.py --algo simnpo_gdr --model_dir muse-bench/MUSE-News_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/news/raw/forget.txt --retain_data_file ../data/news/raw/retain1.txt --out_dir ./ckpt/news/simnpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 4 --beta 0.7 --coeff 0.1 --npo_coeff 1.0

# books
python unlearn.py --algo simnpo_gdr --model_dir muse-bench/MUSE-Books_target --tokenizer_dir meta-llama/Llama-2-7b-hf --data_file ../data/books/raw/forget.txt --retain_data_file ../data/books/raw/retain1.txt --out_dir ./ckpt/books/simnpo_gdr --max_len 2048 --epochs 10 --lr 1e-5 --per_device_batch_size 4 --beta 0.75 --coeff 0.1 --npo_coeff 1.0
```

- `algo`: Unlearning algorithm to run (`simnpo`, `simnpo_gdr`, `ga`, `ga_gdr`, `ga_klr`, `npo`, `npo_gdr`, `npo_klr`, or `tv`).
- `model_dir`: Directory of the target model.
- `tokenizer_dir`: Directory of the tokenizer.
- `data_file`: Forget set.
- `retain_data_file`: Retain set for GDR/KLR regularizations if required by the algorithm.
- `out_dir`: Directory to save the unlearned model (default: `ckpt`).
- `max_len`: Maximum input length (default: 2048).
- `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.

----
**Resulting models are saved in the `ckpt` folder as shown:**
```
ckpt
├── news/
│   ├── simnpo_gdr/
│   │   ├── checkpoint-102
│   │   ├── checkpoint-204
│   │   ├── checkpoint-306
│   │   └── ...
│   └── npo/
│       └── ...
└── books/
    ├── simnpo_gdr
    └── ...
```

# Evaluate the unlearned model

- To evaluate your unlearned model(s), run `eval.py` from the root of this repository with the following command-line arguments:

    - `--model_dirs`: A list of directories containing the unlearned models. These can be either HuggingFace model directories or local storage paths.
    - `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
    - `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
    - `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
    - `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
    - `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
    - `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

- Run the following command with placeholder values:

    ```python
    python eval.py \
    --model_dirs "repo/model1" "repo/model2" \
    --names "model1" "model2" \
    --corpus books \
    --out_file "out.csv"
    ```

- For `News`, we select the result of simnpo from epoch 10. For `Books`, we select the result of simnpo from epoch 10.