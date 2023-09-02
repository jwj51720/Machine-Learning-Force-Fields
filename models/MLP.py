import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, dropout_rate
    ):  # input_size = 3 -> xyz, output_size = 3 -> forces
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),  # hidden_size = 256
            nn.BatchNorm1d(self.hidden_size),  # 배치 정규화, 이는 eval 모드에서는 적용되지 않는다.
            nn.ReLU(),  # 비선형성을 추가하면서도 계산이 빠르고 경사 소실 문제를 완화
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),  # 입력이 음수일 때도 작은 기울기를 가지는 ReLU의 변형
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  #
            nn.ReLU(),  # 비활성함수 이전에 배치정규화를 해주는 것이 일반적.
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, self.output_size),
        )

    def forward(self, x):
        y = self.layers(x)

        return y
