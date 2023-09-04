import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, num_layers):
        super(BiLSTM, self).__init__()

        # Bidirectional LSTM with Dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )  # bidirectional true만 해주면 가능

        # Bidirectional LSTM이므로 hidden_size 조정
        self.linear = nn.Sequential(
            nn.Linear(
                hidden_size * 2, hidden_size
            ),  # lstm을 통과한 hidden size는 bidirectional이기 때문에 2배임
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        energy = self.linear(lstm_out[:, -1, :])  # 역시 마지막 시퀀스만 취함
        return energy
