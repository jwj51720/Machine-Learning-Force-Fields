import pandas as pd
from ase.io import read
from sklearn.model_selection import train_test_split


class Preprocess:
    def __init__(self, config):
        self.config = config
        self.cfg_data = config["data"]
        self.sequence_train = None
        self.sequence_test = None

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

    def preprocessing_force(self, data, is_train=True) -> pd.DataFrame:
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
        if is_train:  # train
            self.sequence_train = sequence_data
        else:  # test
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
            x, y, random_state=self.config["seed"]
        )
        train_df = pd.DataFrame(data={"X": x_train.values.tolist(), "y": y_train})
        valid_df = pd.DataFrame(data={"X": x_valid.values.tolist(), "y": y_valid})
        return train_df, valid_df
