import argparse
import torch
from utils.utils import set_seed, read_json
from dataset.dataset import ForceDataset, get_loader
from dataset.preprocess import Preprocess


def main(config):
    preprocessor = Preprocess(config)
    train_data = preprocessor.load_train_data()
    train_df = preprocessor.preprocessing_force(train_data, is_train=True)
    train_df, valid_df = preprocessor.train_valid_split(train_df)

    train_dataset = ForceDataset(train_df, is_train=True)
    valid_dataset = ForceDataset(valid_df, is_train=False)
    train_loader, valid_loader = get_loader(train_dataset, valid_dataset, config)

    pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="samsung")
    args.add_argument(
        "-c",
        "--config",
        default="./config/base_config.json",
        type=str,
        help="config path",
    )
    args.add_argument("-t", "--task", default="force", type=str, help="force or energy")
    args = args.parse_args()
    config = read_json(args.config)

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["seed"])
    main(config)
