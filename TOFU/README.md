# TOFU and WMDP

## Installation

```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Get the origin model

* For TOFU, we use 8 GPUs and fine-tune the model for 5 epochs with a learning rate of 1e-5 to obtain the origin model. The origin model can be downloaded directly from [here](https://drive.google.com/drive/folders/1L47Hf813gal8RD581S3XrWHnY_0ll4y4?usp=sharing).

* For WMDP, you can obtained it from [here](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta).


## Get and evaluate the unlearned model

* First, you need to update the `model_path` in the `forget.yaml` file to the path corresponding to the origin model. 

* You can also modify the `save_dir` to change the path where the unlearned model will be saved.

* To unlearn a model on a forget set, use the following command:
    ```python
    # forget05
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$master_port forget.py --config-name=forget.yaml split=forget05 npo_coeff=0.1375 beta=2.5

    # forget10
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$master_port forget.py --config-name=forget.yaml split=forget10 npo_coeff=0.125 beta=4.5

    # wmdp
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$master_port forget_wmdp.py --config-name=forget_wmdp.yaml
    ```

* Once the unlearning process is complete, the results will be saved in `${save_dir}/checkpoint/aggregate_stat.txt`.