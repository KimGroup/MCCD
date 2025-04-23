import argparse
import os
import random
import string

def generate_random_string(length=16):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def parse_all_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_distance', type=int, default=3,
                        help='Code distance (used for dataset and model dimensions)')
    parser.add_argument('--logical_circuit_index', type=str, default='4',
                        help='Logical circuit index (must be "4" for this script)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Training batch size')
    parser.add_argument('--if_final_round_syndrome', action='store_true',
                        help='Whether to include final round syndrome input')
    parser.add_argument('--if_large_lstm_2q', action='store_true',
                        help='Use larger LSTM layers for 2Q model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to directory containing pretrained model checkpoint')
    parser.add_argument('--run_index', type=str, default='0',
                        help='Run index for naming checkpoints')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Optional override for model checkpoint path')

    parser.add_argument('--train_data_dir', type=str, default='cached_qec_data/train',
                        help='Path to training dataset directory')
    parser.add_argument('--val_data_dir', type=str, default='cached_qec_data/val',
                        help='Path to validation dataset directory')
    parser.add_argument('--train_depth_list', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='List of depths to use for training')
    parser.add_argument('--val_depth_list', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='List of depths to use for validation')
    parser.add_argument('--save_model_every_n_batches', type=int, default=500,
                        help='Save model checkpoint every N batches')
    parser.add_argument('--validate_every_n_batches', type=int, default=100,
                        help='Run evaluation every N batches')

    args, _ = parser.parse_known_args(argv)
    return args

def generate_save_path(argv=None, mle=False):
    args = parse_all_args(argv)

    if mle:
        path = f"{args.code_distance}_{args.logical_circuit_index}_mle_decoder"
    else:
        if args.model_save_path:
            path = args.model_save_path
        else:
            args.model_save_path = generate_random_string()
            path = args.model_save_path

    print('Model save to:', args.model_save_path, flush=True)
    return path

def generate_data_path(argv=None):
    args = parse_all_args(argv)
    path = f"ml_data/d{args.code_distance}_c{args.logical_circuit_index}"
    return path
