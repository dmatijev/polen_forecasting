import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, drop_prob = 0.1):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        #self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, x, hidden):
        # just run the encoder, for a moment... Will fix that later and attach some decoder to it!

     
        all_hidden_states, hidden = self.lstm(x)#, hidden)
        lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only last hidden state
        #out = self.sigmoid(lstm_out) # not sure we need this
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out
   
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


