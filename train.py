import argparse
import torch
from utils.utils import set_seed, read_json
from dataset.dataset import ForceDataset, get_loader
from dataset.preprocess import Preprocess
from models.get_model import get_models
from trainer.trainer import BaseTrainer
from datetime import datetime
import pytz


def force(config):
    print("=====================start preprocessing====================")
    preprocessor = Preprocess(config)
    print("(1/3) ..load train data..")
    train_data = preprocessor.load_train_data()
    print("(2/3) ..xyz data to df..")
    train_df = preprocessor.data2df(train_data, is_train=True)
    print("(3/3) ..split train and valid..")
    train_df, valid_df = preprocessor.train_valid_split(train_df)

    train_dataset = ForceDataset(train_df, config, is_train=True)
    valid_dataset = ForceDataset(valid_df, config, is_train=True)
    train_loader, valid_loader = get_loader(train_dataset, valid_dataset, config)
    print(f'current device is {config["device"]}')
    print("=======================start training=======================")
    model = get_models(config)
    trainer = BaseTrainer(model, train_loader, valid_loader, config)
    model = trainer.training()
    kst_time = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(
        model.state_dict(),
        f'{trainer.save_dir}/{config["model"]["force"]["type"]}_{kst_time}.pt',
    )
    print("========================training done=======================")


def energy(config):
    print("=====================start preprocessing====================")
    preprocessor = Preprocess(config)
    print("(1/3) ..load train data..")
    train_data = preprocessor.load_train_data()
    print("(2/3) ..xyz data to df..")
    train_df = preprocessor.data2df(train_data, is_train=True)
    print("(3/3) ..load train and test df..")
    train_df, test_df = preprocessor.force_split(train_data)
    preprocessor.make_sequence()


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
    config["task"] = args.task
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config["seed"])
    if config["task"] == "force":
        force(config)
    elif config["task"] == "energy":
        energy(config)
