import pandas as pd
from ase.io import read
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from collections import Counter


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_data = config["data"]
        self.sequence_train = []
        self.sequence_test = []
        self.energies = []

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
        volumes = []
        masses = []
        Ns = []
        Sis = []
        for i in range(len(data)):
            mole = data[i]
            atoms = len(mole)
            sequence_data.append(atoms)

            position = mole.get_positions()  # (n,3)
            force = mole.get_forces()  # label 1, (n,3)
            volume = mole.get_volume()
            mass = mole.get_masses()
            symbol = mole.get_chemical_symbols()
            element_counts = Counter(symbol)

            energy = mole.get_total_energy()  # label 2, (n)
            self.energies.append(energy)

            for j in range(len(mole)):  # jth atom in ith mole
                positions_x.append(position[j][0])
                positions_y.append(position[j][1])
                positions_z.append(position[j][2])
                forces.append(force[j])
                volumes.append(volume)
                masses.append(mass[j])
                Ns.append(element_counts["N"])
                Sis.append(element_counts["Si"])
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
                "volume": volumes,
                "N": Ns,
                "Si": Sis,
                "mass": masses,
                "force": forces,
            }
        )
        return df

    def train_valid_split_f(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        x = df.drop(columns=["force"])
        y = df["force"]
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=self.config["seed"]
        )
        x_train["force"] = y_train
        x_valid["force"] = y_valid
        return x_train, x_valid

    def train_valid_split_e(self, sequences, mask, label):
        (x_train, x_valid, m_train, m_valid, y_train, y_valid) = train_test_split(
            sequences, mask, label, test_size=0.2, random_state=self.config["seed"]
        )
        return [x_train, m_train, y_train], [x_valid, m_valid, y_valid]

    def infenrence_preprocessing_force(self, preds):
        bundles_test = self.take_bundles_test()
        preds_force = []
        for start, end in bundles_test:  # 시작과 끝. 예를 들어 train[0]은 시작 0번부터 끝 47번일 것
            preds_force.append(np.vstack(preds[start:end]))  # 2차원 array로 저장

        sample = self.load_submission_data()
        preds_force_str = [str(arr) for arr in preds_force]
        sample["force"] = preds_force_str
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

    def force_split(self, df, is_train=True):
        if not is_train:
            df = self.load_force_test()
            df["force"] = (
                df["force"]
                .apply(lambda x: [float(val) for val in x.strip("[]").split()])
                .tolist()
            )
        force_df = pd.DataFrame(
            df["force"].values.tolist(), columns=["force_0", "force_1", "force_2"]
        )
        df = pd.concat([df.drop("force", axis=1), force_df], axis=1)
        return df

    def load_force_test(self):
        file_path = self.cfg_data["force_dir"]
        file_name = f'{self.config["force"]["inference"]["file_name"]}_test.csv'  # test csv in force task
        test = pd.read_csv(f"{file_path}/{file_name}")
        return test

    def load_force_submit(self):
        file_path = self.cfg_data["force_dir"]
        file_name = f'{self.config["force"]["inference"]["file_name"]}_submit.csv'  # sumbit csv in force task
        submit = pd.read_csv(f"{file_path}/{file_name}")
        return submit

    def make_sequence(self, df, is_train=True):
        if is_train:
            bundles = self.take_bundles_train()
            self.sequences_train = [
                df.iloc[start:end].values for start, end in bundles
            ]  # different from the previous one
        else:
            bundles = self.take_bundles_test()
            self.sequences_test = [
                df.iloc[start:end].values for start, end in bundles
            ]  # different from the previous one

    def make_padding(self, is_train=True):
        input_size = self.config["energy"]["model"]["input_size"]
        if is_train:
            sequences = self.sequences_train
        else:
            sequences = self.sequences_test
        max_len = max(seq.shape[0] for seq in sequences)
        padded_sequences = [
            np.vstack(
                [
                    seq,
                    np.zeros((max_len - seq.shape[0], input_size)),
                ]
            )
            for seq in sequences
        ]
        padded_array = np.stack(padded_sequences)
        mask = (padded_array != 0).all(axis=2)
        if is_train:
            return padded_array, mask, self.energies
        else:
            return padded_array, mask
