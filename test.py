import argparse
import os

from train_loop import test

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

    parser.add_argument('--dataset_dir', help='Test dataset path', default='dataset', type=str)
    parser.add_argument('--dataset_name', help='Calculate dataset path', default='INRIA_Lytro', type=str)  # HCI_old HCI_new Stanford INRIA_Lytro
    parser.add_argument('--result_dir', help='Root directory for run results', default='LF_MLS', type=str)
    parser.add_argument('--model_path', help='model load path in result dir', default='LF_MLS_x2.pt')
    parser.add_argument('--scale', help='SR scale', default=2)
    parser.add_argument('--device', help='to device cpu or cuda:n', default="cuda:0")
    
    args = parser.parse_args()

    test(**vars(args))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()   

# ----------------------------------------------------------------------------


