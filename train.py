import argparse
import os

from train_loop import train, test

# ----------------------------------------------------------------------------
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='LFSR',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dataset_dir', help='Training dataset path', default='dataset', type=str)
    parser.add_argument('--result_dir', help='Root directory for run results', default='results', type=str)
    parser.add_argument('--load_path', help='model load path', default=None)
    parser.add_argument('--epochs', help='Epochs', default=300, type=int)
    parser.add_argument('--save_term', help='Model save term', default=50)
    parser.add_argument('--scale', help='SR scale', default=2)
    parser.add_argument('--device', help='to device cpu or cuda:n', default="cuda:0")
    
    args = parser.parse_args()

    train(**vars(args))


# ----------------------------------------------------------------------------6
if __name__ == "__main__":
    main()   

# ----------------------------------------------------------------------------


