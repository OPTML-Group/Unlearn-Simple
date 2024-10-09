import sys
import pathlib
BASELINE_PATH = pathlib.Path(__file__).parent.resolve()
sys.path.append(BASELINE_PATH)

from baselines import it_unlearn, tv_unlearn, finetune

import argparse
from os.path import basename, dirname, join as pathjoin


def main():
    args = get_args()
    print(args.out_dir)
    if args.algo == 'kn':
        raise NotImplementedError()

    elif args.algo == 'tv':
        ft_model_dir = pathjoin(dirname(args.out_dir), basename(args.out_dir) + "_ft")
        finetune(
            args.model_dir, args.data_file, ft_model_dir,
            epochs=args.epochs,
            per_device_batch_size=args.per_device_batch_size,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir
        )
        tv_unlearn(
            args.model_dir, args.out_dir,
            some_pt_model_dir=args.model_dir,
            some_ft_model_dir=ft_model_dir,
            alpha=args.alpha
        )

    else:
        it_unlearn(
            args.model_dir, args.data_file, args.out_dir,
            retain_data_file=args.retain_data_file,
            loss_type=args.algo,
            per_device_batch_size=args.per_device_batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            max_len=args.max_len,
            tokenizer_dir=args.tokenizer_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            beta=args.beta,
            coeff=args.coeff,
            npo_coeff=args.npo_coeff,
            gamma=args.gamma
        )

    return


def get_args():
    parser = argparse.ArgumentParser(description="Unlearning baselines")
    parser.add_argument('--algo', type=str)
    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )
    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )
    parser.add_argument(
        '--resume_from_checkpoint', action='store_true',
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument(
        '--retain_data_file', type=str, default=None,
        help="Path to the retain set file. Required if algo is gradient difference (gd)."
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    # Task vector
    parser.add_argument(
        '--alpha', type=float, default=1.0,
        help="Scaling coefficient scales the task vector if algo is task vector (tv)."
    )

    parser.add_argument(
        '--beta', type=float, default=0.1,
        help="for npo"
    )
    
    parser.add_argument(
        '--coeff', type=float, default=0.1,
        help="for retain loss"
    )

    parser.add_argument(
        '--npo_coeff', type=float, default=0.1,
        help="for forget loss"
    )

    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help="for simnpo"
    )

    args = parser.parse_args()

    if args.algo == 'gd':
        # gradient difference. Retain set is required
        assert args.retain_data_file is not None, "Gradient difference selected. Retain set required."

    if args.resume_from_checkpoint:
        assert args.algo not in {'tv'}, "Cannot resume from checkpoint if the method is task vector."

    return args


if __name__ == '__main__':
    main()
