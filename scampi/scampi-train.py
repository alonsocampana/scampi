import argparse
import os
from utils import get_affix
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for scampi")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--fraction",
         type = float,
         default=1,
         required=False,
         help="Fraction of training data used at each iteration")
    parser.add_argument(
        "--dataset",
         type = str,
         default="biosnap",
         required=False,
         help="Dataset used for training. Either biosnap or stitch.")
    parser.add_argument(
        "--no_transformer",
        action="store_true",
        default=False,
        help="Removes transformer encoder from model")
    parser.add_argument(
        "--no_attn",
        action="store_true",
        default=False,
        help="Replaces attention pooling by average pooling")
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        default=False,
        help="Starts for a given model training from scratch. Otherwise it resumes training")
    parser.add_argument(
        "--permute",
        action="store_true",
        default=False,
        help="Permutes the proteins found in the training and test data")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enables warnings")

    args, _ = parser.parse_known_args()
    os.environ["SCAMPI_FORCE_RETRAIN"] = str(args.force_retrain)
    os.environ["SCAMPI_FRACTION"] = str(args.fraction)
    os.environ["SCAMPI_NO_ATTN"] = str(args.no_attn)
    os.environ["SCAMPI_NO_TRANS"] = str(args.no_transformer)
    os.environ["SCAMPI_PERMUTE"] = str(args.permute)
    os.environ["SCAMPI_CUDA_IS_ENABLED"] = str(args.cuda)
    os.environ["SCAMPI_PATH"] = os.path.join(os.getcwd(), os.pardir)
    if args.verbose:
        pass
    else:
        warnings.filterwarnings("ignore")
    if args.dataset == "biosnap":
        from script_snap import train_model
    else:
        from script_stitch import train_model
    RUN = get_affix(args.dataset,
                     fraction= args.fraction,
                     permute=args.permute,
                     no_attn=args.no_attn,
                     no_trans=args.no_transformer)
    train_model(RUN)
