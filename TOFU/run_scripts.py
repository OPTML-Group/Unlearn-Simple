import os
import random
import shutil
import time
from calendar import c

def run_commands(commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if len(commands) == 0:
        return
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
    os.makedirs(dir, exist_ok=True)

    fout = open("stop_{}.sh".format(dir), "w")
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    sh_path = os.path.join(dir, "run.sh")
    fout = open(sh_path, "w")
    for com in commands:
        print(com, file=fout)
    fout.close()
    if call:
        os.system("bash {}&".format(sh_path))
        time.sleep(delay)


def gen_commands_simnpo_tofu_forget05():

    commands = []
    for beta in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        for gamma in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            command = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18755 forget.py --config-name=forget.yaml forget_loss=simnpo_grad_diff beta={beta} gamma={gamma} grad_diff_coeff=0.1375 num_epochs=10 split=forget05 retain_set=retain95"
            commands.append(command)

    for beta in [2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875]:
        for gamma in [0.0]:
            command = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18755 forget.py --config-name=forget.yaml forget_loss=simnpo beta={beta} gamma={gamma} grad_diff_coeff=0.1375 num_epochs=10 split=forget05 retain_set=retain95"
            commands.append(command)

    return commands


def gen_commands_simnpo_tofu_forget10():

    commands = []
    for beta in [12.5, 15.0, 10.0]:
        for gamma in [0.0]:
            command = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=18756 forget.py --config-name=forget.yaml forget_loss=simnpo_grad_diff beta={beta} gamma={gamma} grad_diff_coeff=0.1 num_epochs=10 split=forget10 retain_set=retain90"
            commands.append(command)

    return commands


def gen_commands_simnpo_wmdp():
    commands = []

    # NPO_RT
    for lr in [3e-06, 3.5e-06]:
        for beta in [5,5, 6.0]:
            for grad_diff_coeff in [5.0]:
                command = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=19764 forget_wmdp.py --config-name=forget_wmdp.yaml forget_loss=simnpo_grad_diff lr={lr} beta={beta} grad_diff_coeff={grad_diff_coeff}"
                commands.append(command)

    return commands


def gen_commands_eval_wmdp():
    commands = []
    for lr in [5e-06]:
        for beta in [0.1]:
            for grad_diff_coeff in [5.0]:
                for step in [125]:
                    for task in ["wmdp_bio", "wmdp_cyber", "mmlu"]:
                        model_path = f"/egr/research-optml/chongyu/NEW-BLUE/TOFU/wmdp_models/origin/unlearned/8GPU_npo_grad_diff_{lr}_wmdp_epoch10_batch4_accum1_beta{beta}_grad_diff_coeff{grad_diff_coeff}_reffine_tuned_evalsteps_per_epoch_seed1001_1/checkpoint-{step}"
                        command = f"torchrun --nproc-per-node=8 --master_port=13769 --no-python lm_eval --model hf --model_args pretrained={model_path} --output_path wmdp_models/result --tasks {task} --batch_size auto"
                        commands.append(command)

    return commands


if __name__ == "__main__":

    commands = gen_commands_simnpo_tofu_forget05()
    print(len(commands))
    run_commands(commands, call=True, dir="gen_commands_simnpo_tofu_forget05", shuffle=False, delay=0.5)