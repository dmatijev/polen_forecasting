import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#import numpy as np

# ugly to have it as globals, but don't care for now... 
input_dim = 5 # input feature vector size
hidden_dim = 10 # size of a hidden vector
n_layers = 1 # number of lstm layers

batch_size = 1
seq_len = 3 # input sequence lenght
epochs = 2

lr=0.005


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self, drop_prob = 0.0):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout = drop_prob, batch_first=True)
        
    def forward(self, x, hidden):
        # just run the encoder, for a moment... Will fix that later and attach some decoder to it!
        lstm_out, hidden = self.lstm(x, hidden)
        return lstm_out
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_().to(device),
                      weight.new(n_layers, batch_size, hidden_dim).zero_().to(device))
        return hidden



def train(model, train_loader, loss_fn, optimizer):

    model.train()
    clip = 5
    for i in range(epochs):
        h = model.init_hidden(batch_size)
        
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            model.zero_grad()
            output, h = model(inputs, h)
            loss = loss_fn(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            # TODO ongoing work... 



if __name__ == "__main__":
    

    # generate dummy input for testing 
    #inp = torch.randn(batch_size, seq_len, input_dim)
    #hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
    #cell_state = torch.randn(n_layers, batch_size, hidden_dim)
    #hidden = (hidden_state, cell_state)
    #(out, hidden) = lstm_layer(inp, hidden)
    
    # Zamolit Slobu da mi priredi train_data, val_data i test_data (TODO)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    
        
    model = Net()
    model.to(device)
    loss = nn.BCELoss() # not sure whether this loss will work for our problem

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    
    train(model, train_loader, loss, optimizer)



















