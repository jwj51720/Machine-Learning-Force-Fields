import argparse
from utils.utils import set_seed, read_json
import torch
from dataset.preprocess import Preprocess
from dataset.dataset import ForceDataset
from torch.utils.data import DataLoader
from models.get_model import get_models
from tqdm import tqdm
import os


def main(config):
    preds = []
    print(f'inference for {config["inference"]["pt_file"]}.pt')
    print("=======================start inference======================")
    preprocessor = Preprocess(config)
    test_data = preprocessor.load_test_data()
    test_df = preprocessor.preprocessing_force(test_data, is_train=False)
    temp_dir = config["data"]["temp_dir"]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    test_dataset = ForceDataset(test_df, config, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config["inference"]["batch"],
    )
    model = get_models(config)
    model.load_state_dict(
        torch.load(f'{config["data"]["save_dir"]}/{config["inference"]["pt_file"]}.pt')
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
    test_df.to_csv(f'{temp_dir}/{config["inference"]["pt_file"]}_test.csv')
    submit.to_csv(f'{temp_dir}/{config["inference"]["pt_file"]}_submit.csv')
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
    main(config)
