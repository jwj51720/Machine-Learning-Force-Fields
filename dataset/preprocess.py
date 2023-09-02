import pandas as pd
from ase.io import read
from sklearn.model_selection import train_test_split
import numpy as np


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_data = config["data"]
        self.sequence_train = []
        self.sequence_test = []

    def load_train_data(self):
        train = read(
            f"{self.cfg_data['data_dir']}/train.xyz", format="extxyz", index=":"
        )
        return train

    def load_test_data(self):
        test = read(f"{self.cfg_data['data_dir']}/test.xyz", format="extxyz", index=":")
        return test

    def load_submission_data(self):
        submit = pd.read_csv(f"{self.cfg_data['data_dir']}/sample_submission.csv")
        return submit

    def data2df(self, data, is_train=True) -> pd.DataFrame:
        (
            sequence_data,
            positions_x,
            positions_y,
            positions_z,
            forces,
        ) = ([], [], [], [], [])
        for i in range(len(data)):
            mole = data[i]
            atoms = len(mole)
            sequence_data.append(atoms)

            position = mole.get_positions()  # (n,3)
            force = mole.get_forces()  # label 1, (n,3)

            for j in range(len(mole)):  # jth atom in ith mole
                positions_x.append(position[j][0])
                positions_y.append(position[j][1])
                positions_z.append(position[j][2])
                forces.append(force[j])
        if is_train:
            self.sequence_train = sequence_data
        else:
            self.sequence_test = sequence_data
            forces = None
        df = pd.DataFrame(
            {
                "position_x": positions_x,
                "position_y": positions_y,
                "position_z": positions_z,
                "force": forces,
            }
        )
        return df

    def train_valid_split(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        x = df.drop(columns=["force"])
        y = df["force"]
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=self.config["seed"]
        )
        x_train["force"] = y_train
        x_valid["force"] = y_valid
        return x_train, x_valid

    def infenrence_preprocessing_force(self, preds):
        bundles_test = self.take_bundles_test()
        preds_force = []
        for start, end in bundles_test:  # 시작과 끝. 예를 들어 train[0]은 시작 0번부터 끝 47번일 것
            preds_force.append(np.vstack(preds[start:end]))  # 2차원 array로 저장

        sample = self.load_submission_data()
        sample["force"] = preds_force
        return sample

    def take_bundles_train(self):
        bundles_train = []
        flag = 0
        if not self.sequence_train:
            train = self.load_train_data()
            for i in range(len(train)):
                mole = train[i]
                atoms = len(mole)
                self.sequence_train.append(atoms)

        for size in self.sequence_train:
            bundles_train.append((flag, flag + size))
            flag += size
        return bundles_train

    def take_bundles_test(self):
        bundles_test = []
        flag = 0
        if not self.sequence_test:
            test = self.load_test_data()
            for i in range(len(test)):
                mole = test[i]
                atoms = len(mole)
                self.sequence_test.append(atoms)

        for size in self.sequence_test:
            bundles_test.append((flag, flag + size))
            flag += size
        return bundles_test
