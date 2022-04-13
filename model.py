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
    #def __init__(self, input_dim, hidden_dim, hidden_dim2, seq_len, nr_days = 1, n_layers=1, attention = False):
    def __init__(self, input_dim, hidden_dim, seq_len, nr_days = 1, n_layers=1, attention = False, device = torch.device("cpu")):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, bias=True, batch_first=True)
        #self.fc1 = nn.Linear(hidden_dim, hidden_dim2)
        #self.fc = nn.Linear(hidden_dim2, 1)
        self.fc = nn.Linear(hidden_dim, 1)
        #self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim2)
        self.lstm_cell = nn.LSTMCell(6, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        #self.hidden_dim2 = hidden_dim2
        self.use_attention_layer = attention
        self.attention = Attention(seq_len)
        self.nr_days = nr_days
        self.device = device
        
    def forward(self, x, meteo):# hidden):    
        all_hidden_states, hidden = self.lstm(x)#, hidden)

        if self.use_attention_layer:
            encoder_out = self.attention(all_hidden_states)
            encoder_out = encoder_out.contiguous().view(encoder_out.shape[0], -1)
        else:
            encoder_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only the last hidden state
            
        
        #out = self.fc(self.sigmoid(self.fc1(encoder_out))) # old approach
        #return out
    
        #hx = torch.zeros(encoder_out.shape[0], self.hidden_dim2).to('cuda')
        #cx = torch.zeros(encoder_out.shape[0], self.hidden_dim2).to('cuda')
        
        hx = encoder_out
        cx = torch.zeros(encoder_out.shape).to(self.device)
        out = []
        for i in range(self.nr_days): # start decoding
            #hx, cx = self.lstm_cell(encoder_out, (hx, cx))
            hx, cx = self.lstm_cell(meteo[:,i,:-1], (hx, cx))
            out.append(self.fc(hx))

        return torch.cat(out, dim = 1)
   
#    def init_hidden(self, batch_size, device = torch.device("cpu")):
#        weight = next(self.parameters()).data
#        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#        return hidden


