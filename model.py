import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.v = nn.Linear(input_dim, 1) # linear combination coefficients (i.e. attention weights)
        self.input_dim = input_dim

    def forward(self, encoder_states):
        states_reshaped = torch.swapaxes(encoder_states, 1, 2)
        return self.v(states_reshaped)
        


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, n_layers=1, attention = False):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, bias=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_attention_layer = attention
        self.attention = Attention(seq_len)
        
    def forward(self, x):# hidden):    
        all_hidden_states, hidden = self.lstm(x)#, hidden)

        if self.use_attention_layer:
            lstm_out = self.attention(all_hidden_states)
            lstm_out = lstm_out.contiguous().view(lstm_out.shape[0], -1)
        else:
            lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only the last hidden state
            
        out = self.fc(self.sigmoid(self.fc1(lstm_out)))
        
        return out
   
#    def init_hidden(self, batch_size, device = torch.device("cpu")):
#        weight = next(self.parameters()).data
#        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#        return hidden


