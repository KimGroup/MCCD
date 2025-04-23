import torch
import torch.nn as nn

class LSTMReLUUnit(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMReLUUnit, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out_last = lstm_out[:, -1, :]
        return lstm_out_last, hidden

class TwoQubitLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(TwoQubitLSTM, self).__init__()
        self.lstm2 = nn.LSTM(input_size * 2, hidden_size * 2, num_layers, batch_first=True)

    def forward(self, x1, x2, hidden1, hidden2):
        x = torch.cat((x1, x2), dim=1).unsqueeze(1)
        hidden = (torch.cat((hidden1[0], hidden2[0]), dim=2),
                  torch.cat((hidden1[1], hidden2[1]), dim=2))
        lstm_out, hidden = self.lstm2(x, hidden)
        lstm_out_last = lstm_out[:, -1, :]
        out1, out2 = torch.split(lstm_out_last, lstm_out_last.size(-1) // 2, dim=1)
        h, c = hidden
        h1, h2 = torch.split(h, h.size(-1) // 2, dim=2)
        c1, c2 = torch.split(c, c.size(-1) // 2, dim=2)
        hidden1 = (h1.contiguous(), c1.contiguous())
        hidden2 = (h2.contiguous(), c2.contiguous())
        return (out1, hidden1), (out2, hidden2)

class TwoQubitLSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(TwoQubitLSTM2, self).__init__()
        self.lstm_tgt = nn.LSTM(input_size * 2, hidden_size, num_layers, batch_first=True)
        self.lstm_ctrl = nn.LSTM(input_size * 2, hidden_size, num_layers, batch_first=True)

    def forward(self, x1, x2, hidden1, hidden2):
        x = torch.cat((x1, x2), dim=1).unsqueeze(1)
        lstm_out_tgt, hidden_tgt = self.lstm_tgt(x, hidden1)
        lstm_out_ctrl, hidden_ctrl = self.lstm_ctrl(x, hidden2)
        return (lstm_out_tgt[:, -1, :], hidden_tgt), (lstm_out_ctrl[:, -1, :], hidden_ctrl)

class GateDependentLSTMReLUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(GateDependentLSTMReLUModel, self).__init__()
        self.lstm_dict = nn.ModuleDict()
        for gate in ['I', 'X', 'Y', 'Z', 'H']:
            self.lstm_dict[gate] = LSTMReLUUnit(input_size, hidden_size, num_layers)
        self.relu = nn.ReLU()

    def forward(self, x, circuit):
        num_logical_qubits = x.size(2)
        hidden_arr = [[None] for _ in range(num_logical_qubits)]
        for gate, t, q_ind_list in circuit:
            for q in q_ind_list:
                last_hidden_q = hidden_arr[q][-1]
                x_t_q = x[:, t, q, :]
                _, hidden = self.lstm_dict[gate](x_t_q, last_hidden_q)
                hidden_arr[q].append(hidden)
        last_outs = [self.relu(hidden_arr[q][-1][0][-1, :, :]).unsqueeze(1)
                     for q in range(num_logical_qubits)]
        return torch.cat(last_outs, dim=1)

class GateDependentLSTMReLUModelw2Q(GateDependentLSTMReLUModel):
    def __init__(self, input_size, hidden_size, num_layers=2, if_large_lstm_2q=True):
        super().__init__(input_size, hidden_size, num_layers)
        if if_large_lstm_2q:
            self.two_qubit_lstm = TwoQubitLSTM(input_size, hidden_size, num_layers)
        else:
            self.two_qubit_lstm = TwoQubitLSTM2(input_size, hidden_size, num_layers)

    def forward(self, x, circuit):
        num_logical_qubits = x.size(2)
        hidden_arr = [[None] for _ in range(num_logical_qubits)]
        for gate, t, q_ind_list in circuit:
            if gate in ['CX', 'CNOT']:
                for i in range(0, len(q_ind_list), 2):
                    q1, q2 = q_ind_list[i], q_ind_list[i + 1]
                    x_t_q1 = x[:, t, q1, :]
                    x_t_q2 = x[:, t, q2, :]
                    last_hidden_q1 = hidden_arr[q1][-1]
                    last_hidden_q2 = hidden_arr[q2][-1]
                    (out1, h1), (out2, h2) = self.two_qubit_lstm(x_t_q1, x_t_q2, last_hidden_q1, last_hidden_q2)
                    hidden_arr[q1].append(h1)
                    hidden_arr[q2].append(h2)
            else:
                for q in q_ind_list:
                    x_t_q = x[:, t, q, :]
                    _, hidden = self.lstm_dict[gate](x_t_q, hidden_arr[q][-1])
                    hidden_arr[q].append(hidden)
        last_outs = [self.relu(hidden_arr[q][-1][0][-1, :, :]).unsqueeze(1)
                     for q in range(num_logical_qubits)]
        return torch.cat(last_outs, dim=1)

class MainReadout(nn.Module):
    def __init__(self, hidden_size, fx_len):
        super(MainReadout, self).__init__()
        self.fc = nn.Linear(hidden_size + fx_len, hidden_size + fx_len)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size + fx_len, 2)

    def forward(self, x, fx):
        x = torch.cat((x, fx), dim=1)
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class AuxiliaryReadout(nn.Module):
    def __init__(self, hidden_size):
        super(AuxiliaryReadout, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MultiQubitMainReadout(nn.Module):
    def __init__(self, hidden_size, fx_len):
        super(MultiQubitMainReadout, self).__init__()
        self.main_readout = MainReadout(hidden_size, fx_len)

    def forward(self, x, fx):
        return torch.cat([self.main_readout(x[:, q, :], fx[:, q, :]).unsqueeze(1)
                          for q in range(x.size(1))], dim=1)

class MultiQubitAuxiliaryReadout(nn.Module):
    def __init__(self, hidden_size):
        super(MultiQubitAuxiliaryReadout, self).__init__()
        self.auxiliary_readout = AuxiliaryReadout(hidden_size)

    def forward(self, x):
        return torch.cat([self.auxiliary_readout(x[:, q, :]).unsqueeze(1)
                          for q in range(x.size(1))], dim=1)

class CircuitLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(CircuitLSTMDecoder, self).__init__()
        self.gate_dependent_lstm_relu = GateDependentLSTMReLUModel(input_size, hidden_size, num_layers)
        fx_len = input_size // 2
        self.main_readout = MultiQubitMainReadout(hidden_size, fx_len)
        self.auxiliary_readout = MultiQubitAuxiliaryReadout(hidden_size)

    def forward(self, x, circuit, fx):
        x = self.gate_dependent_lstm_relu(x, circuit)
        main_out = self.main_readout(x, fx)
        auxiliary_out = self.auxiliary_readout(x)
        return main_out, auxiliary_out

class CircuitLSTMDecoderw2Q(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, if_large_lstm_2q=True):
        super(CircuitLSTMDecoderw2Q, self).__init__()
        self.gate_dependent_lstm_relu = GateDependentLSTMReLUModelw2Q(input_size, hidden_size, num_layers, if_large_lstm_2q)
        fx_len = input_size // 2
        self.main_readout = MultiQubitMainReadout(hidden_size, fx_len)
        self.auxiliary_readout = MultiQubitAuxiliaryReadout(hidden_size)

    def forward(self, x, circuit, fx):
        x = self.gate_dependent_lstm_relu(x, circuit)
        main_out = self.main_readout(x, fx)
        auxiliary_out = self.auxiliary_readout(x)
        return main_out, auxiliary_out

    def init_from(self, model):
        self.main_readout.load_state_dict(model.main_readout.state_dict())
        self.auxiliary_readout.load_state_dict(model.auxiliary_readout.state_dict())
        self.gate_dependent_lstm_relu.load_state_dict(model.gate_dependent_lstm_relu.state_dict(), strict=False)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.gate_dependent_lstm_relu.two_qubit_lstm.parameters():
            param.requires_grad = True
