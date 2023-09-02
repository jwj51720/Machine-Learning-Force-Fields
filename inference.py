import argparse
from utils.utils import set_seed, read_json
import torch
from dataset.preprocess import Preprocess
from dataset.dataset import ForceDataset
from torch.utils.data import DataLoader
from models.get_model import get_models
from tqdm import tqdm


def main(config):
    preds = []
    print("=======================start inference======================")
    preprocessor = Preprocess(config)
    test_data = preprocessor.load_test_data()
    test_df = preprocessor.preprocessing_force(test_data, is_train=True)

    test_dataset = ForceDataset(test_df, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
    )
    model = get_models(config)
    model.load_state_dict(
        torch.load(f'{config["trainer"]["save_dir"]/config["trainer"]["save_name"]}')
    )
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(test_loader):
            output = model(inputs)
            pred = output.detach().cpu().numpy()
            preds.extend(pred)
    print("========================inference done======================")
    print("..postprocessing..")
    test_df["force"] = preds
    submit = preprocessor.infenrence_preprocessing_force(preds)
    test_df.to_csv(f'{config["data"]["data_dir"]}/"energy_test.csv"')
    submit.to_csv(f'{config["trainer"]["result"]}/"submission.csv"')


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