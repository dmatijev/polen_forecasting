import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim) # linear combination coefficients (i.e. attention weights)
        self.W2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.hidden_dim = hidden_dim

    def forward(self, hx, encoder_states): # dimension 
        att_weights = torch.bmm(encoder_states, self.W1(hx).unsqueeze(2)).squeeze(2)      
        att_weights = self.softmax(att_weights)
        at = torch.bmm(att_weights.unsqueeze(1), encoder_states)
        ut = torch.cat((at.squeeze(1), hx), dim = 1)
        vt = self.W2(ut)
        ot = self.dropout(self.tanh(vt))
        
        return att_weights, ot
        
    


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, nr_days = 1, n_layers=1, dropout = 0.25, attention = False, device = torch.device("cpu")):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.lstm_cell = nn.LSTMCell(6, hidden_dim) # 6 different meteo data
        self.sigmoid = nn.Sigmoid()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_attention_layer = attention
        self.attention = Attention(hidden_dim, dropout)
        self.nr_days = nr_days
        self.device = device
        self.input_dim = input_dim
        
    def forward(self, x, meteo, shared_weights = False):# hidden):    
        
 
        all_encoder_states, hidden = self.lstm(x)#, hidden)

        encoder_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only the last hidden state
        cx = hidden[1].contiguous().view(-1, self.hidden_dim)    

        if shared_weights:
            encoder_out = torch.mean(encoder_out, 0, True)
            cx = torch.mean(cx, 0, True)
            all_encoder_states = torch.mean(all_encoder_states, 0, True) 
            
        
        hx = encoder_out

        #cx = torch.zeros(encoder_out.shape).to(self.device)
        out = []
        for i in range(self.nr_days): # start decoding
            hx, cx = self.lstm_cell(meteo[:,i,:], (hx, cx))
            ot = hx
            if self.use_attention_layer:
                att_weights, ot = self.attention(hx, all_encoder_states) # TODO IMPLEMENT THIS

            out.append(self.fc(ot))

        return torch.cat(out, dim = 1)
 
"""

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.v = nn.Linear(input_dim, 1) # linear combination coefficients (i.e. attention weights)
        self.input_dim = input_dim

    def forward(self, encoder_states):
        states_reshaped = torch.swapaxes(encoder_states, 1, 2)
        return self.v(states_reshaped)
        
    


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, nr_days = 1, n_layers=1, attention = False, device = torch.device("cpu")):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.lstm_cell = nn.LSTMCell(6, hidden_dim) # 6 different meteo data
        self.sigmoid = nn.Sigmoid()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_attention_layer = attention
        self.attention = Attention(seq_len)
        self.nr_days = nr_days
        self.device = device
        self.input_dim = input_dim
        
    def forward(self, x, meteo, shared_weights = False):# hidden):    

        all_hidden_states, hidden = self.lstm(x)#, hidden)

        if self.use_attention_layer:
            encoder_out = self.attention(all_hidden_states)
            encoder_out = encoder_out.contiguous().view(encoder_out.shape[0], -1)
        else:
            encoder_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only the last hidden state
            

        if shared_weights:
            encoder_out = torch.mean(encoder_out, 0, True)
        
        hx = encoder_out
        cx = torch.zeros(encoder_out.shape).to(self.device)
        out = []
        for i in range(self.nr_days): # start decoding
            #hx, cx = self.lstm_cell(encoder_out, (hx, cx))
            hx, cx = self.lstm_cell(meteo[:,i,:], (hx, cx))
            out.append(self.fc(hx))

        return torch.cat(out, dim = 1)
 
"""
