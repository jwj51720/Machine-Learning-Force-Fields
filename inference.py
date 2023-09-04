import argparse
from utils.utils import set_seed, read_json
import torch
from dataset.preprocess import Preprocess
from dataset.dataset import ForceDataset, EnergyDataset
from torch.utils.data import DataLoader
from models.get_model import get_models
from tqdm import tqdm
import numpy as np
import os
import sys


def force(config):
    np.set_printoptions(threshold=sys.maxsize)
    preds = []
    print(f'inference for {config["force"]["inference"]["file_name"]}.pt')
    print("=======================start inference======================")
    preprocessor = Preprocess(config)
    print("(1/3) ..load test data..")
    test_data = preprocessor.load_test_data()
    print("(2/3) ..xyz data to df..")
    test_df = preprocessor.data2df(test_data, is_train=False)
    force_dir = config["data"]["force_dir"]

    test_dataset = ForceDataset(test_df, config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config["force"]["inference"]["batch"],
    )
    model = get_models(config)
    model.load_state_dict(
        torch.load(f'{force_dir}/{config["force"]["inference"]["file_name"]}.pt')
    )
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            inputs = inputs.to(config["device"])
            output = model(inputs)
            pred = output.detach().cpu().numpy()
            preds.extend(pred)
    print("(3/3) ..postprocessing..")
    test_df["force"] = preds

    submit = preprocessor.infenrence_preprocessing_force(preds)
    test_df.to_csv(
        f'{force_dir}/{config["force"]["inference"]["file_name"]}_test.csv', index=False
    )
    submit.to_csv(
        f'{force_dir}/{config["force"]["inference"]["file_name"]}_submit.csv',
        index=False,
    )
    print(f'..save {force_dir}/{config["force"]["inference"]["file_name"]}_test.csv')
    print(f'..save {force_dir}/{config["force"]["inference"]["file_name"]}_submit.csv')
    print("========================inference done======================")


def energy(config):
    preds = []
    print(f'inference for {config["energy"]["inference"]["file_name"]}.pt')
    print("=======================start inference======================")
    preprocessor = Preprocess(config)
    print("(1/4) ..load test df and force split..")
    test_df = preprocessor.force_split(None, is_train=False)
    print("(2/4) ..make sequence..")
    preprocessor.make_sequence(test_df, is_train=False)
    print("(3/4) ..make padding..")
    test, test_mask = preprocessor.make_padding(is_train=False)

    test_dataset = EnergyDataset([test, test_mask], is_train=False)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config["energy"]["inference"]["batch"],
    )
    model = get_models(config)
    model.load_state_dict(
        torch.load(
            f'{config["data"]["energy_dir"]}/{config["energy"]["inference"]["file_name"]}.pt'
        )
    )
    model.eval()
    with torch.no_grad():
        for inputs, masks in tqdm(test_loader):
            inputs = inputs.to(config["device"])
            output = model(inputs)
            pred = output.detach().cpu().numpy()
            preds.extend(pred)
    print("(4/4) ..postprocessing..")
    preds = [pred.item() for pred in preds]
    sample = preprocessor.load_force_submit()
    sample["energy"] = preds
    sample.to_csv(
        f'{config["data"]["energy_dir"]}/{config["energy"]["inference"]["file_name"]}.csv',
        index=False,
    )
    print(
        f'..save {config["data"]["energy_dir"]}/{config["energy"]["inference"]["file_name"]}.csv'
    )
    print("========================inference done======================")


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
    if config["task"] == "force":
        force(config)
    elif config["task"] == "energy":
        energy(config)
