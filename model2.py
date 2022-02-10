import torch
import torch.nn as nn


class Attention2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.v = nn.Linear(input_dim, 1) # linear combination coefficients (i.e. attention weights)
        self.input_dim = input_dim

    def forward(self, encoder_states):
        states_reshaped = torch.swapaxes(encoder_states, 1, 2)
        return self.v(states_reshaped)
        

class Net2(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim2, seq_len, n_layers=1, attention = False):
        super(Net2, self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers = n_layers, bias=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_attention_layer = attention
        self.attention = Attention2(seq_len)
        self.lstmCell = nn.LSTMCell(input_size = input_dim, hidden_size = hidden_dim) #ovdje ne mora biti input_dim, vjerojatno je input_dim-1 jer cemo nove podatke imati bez peludi, samo prognoza
        
    def decoder(self, x, h, k):
        #hx = torch.randn(x.size()[0], self.hidden_dim) # (batch, hidden_size)
        hx = h
        #print('h:', h.size())
        #print('x',x[:,1,:].size())
        cx = torch.randn(x.size()[0], self.hidden_dim) #ovaj hidden dim vjerojatno moze biti drukciji od hidden_dim i hidden_dim2
        cx = cx.to("cuda") #ovo treba popraviti da ne bude hardcoded nego da se negdje proslijedi modelu ili nesto
        output = []
        #for i in range(x.size()[0]):
        for i in range(k):  
            hx, cx = self.lstmCell(x[:,i,:], (hx, cx)) #nisam sigurna da ce ovaj x[i] dobro funkcionirati, hoce li x[i] imati k unosa?
            output.append(hx)
        output = torch.stack(output, dim=0)
        #print('output:',output.size())
        return output
        
        
    def forward(self, x):# hidden):
        #print('x: ',x.size())
        #print(x[1][0])
        #print(x.size()[0])
        all_hidden_states, hidden = self.lstm(x)#, hidden)

        if self.use_attention_layer:
            lstm_out = self.attention(all_hidden_states)
            lstm_out = lstm_out.contiguous().view(lstm_out.shape[0], -1)
        else:
            lstm_out = hidden[0].contiguous().view(-1, self.hidden_dim) # take only the last hidden state
            
        #ovdje staviti decoder uz k=1, uzeti i-ti output (i ide od 1 do k) i onda ga pustiti kroz ove slojeve da se dobije k predikcija
        out = []
        outDecode = self.decoder(x,lstm_out,1)
        #print('outDecode',outDecode.size())
        for el in outDecode:
            out.append(self.fc(self.sigmoid(self.fc1(el))))
        out = torch.stack(out, dim=0)
        #print('out: ',out.size())
        return out
   
#    def init_hidden(self, batch_size, device = torch.device("cpu")):
#        weight = next(self.parameters()).data
#        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#        return hidden


