# WMDP

## Installation

To create a conda environment for Python 3.9, run:
```bash
conda env create -f environment.yml
conda activate wmdp
```

## Get the data
Follow the [link](https://github.com/centerforaisafety/wmdp?tab=readme-ov-file) to download the WMDP-Bio dataset and place it in the `./WMDP/files/data`.

## Get the unlearned model
1. Run the command `bash run_wmdp_unlearn.sh`.
2. After the command is complete, the checkpoints and results will be stored in `./WMDP/files/results/unlearn_wmdp_bio/SimNPO`.