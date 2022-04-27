import paddle.nn as nn
import paddle.nn.functional as F


class AnglePredict(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=None, rnn_style="lstm"):
        super().__init__()
        self.rnn_style = rnn_style.lower()
        if self.rnn_style == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        elif self.rnn_style == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Please input right rnn style, like [gru, lstm]")

        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, angle_seq):
        rnn_outputs, _ = self.rnn(angle_seq)
        cls_state = rnn_outputs[:, -1]
        cls_drop = self.dropout(cls_state)
        logits = F.sigmoid(self.classifier(cls_drop))
        logits = logits.squeeze()
        
        return logits

