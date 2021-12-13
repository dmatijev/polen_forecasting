import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_test_split import load_data
from data_set import Dataset
#import numpy as np

# ugly to have it as globals, but don't care for now... 
#input_dim = 5 # input feature vector size
hidden_dim = 100 # size of a hidden vector
n_layers = 1 # number of lstm layers

batch_size = 1
seq_len = 7 # input sequence lenght
epochs = 400

lr_rate=0.000001


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Net(nn.Module):
    def __init__(self, input_dim, drop_prob = 0.1):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        #self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x, hidden):
        # just run the encoder, for a moment... Will fix that later and attach some decoder to it!

     
        all_hidden_states, hidden = self.lstm(x)#, hidden)
        lstm_out = hidden[0].contiguous().view(-1, hidden_dim) # take only last hidden state
        #out = self.sigmoid(lstm_out) # not sure we need this
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        return out
        
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(n_layers, batch_size, hidden_dim).zero_().to(device),
                      weight.new(n_layers, batch_size, hidden_dim).zero_().to(device))
        return hidden



def train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer):

    
    clip = 5
    for i in range(epochs):
        train_epoch_loss = 0
        
        model.train()
        h = model.init_hidden(batch_size)
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            model.zero_grad()
            
            output = model(inputs, h)
            print(inputs)
            print(labels)
            print(output)
            print()

            loss = loss_fn(output.squeeze(), labels.squeeze())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_epoch_loss += loss.detach()
        print(f"Epoch {i}: total TRAIN loss: {train_epoch_loss/len(train_loader)}")
        
        # validate the current model
        model.eval()       
        valid_epoch_loss = 0
        for (inputs, labels) in valid_loader:

            inputs, labels = inputs.to(device), labels.to(device)            
            output = model(inputs, h)
            #print(inputs)
            #print(labels)
            #print(output)
            loss = loss_fn(output.squeeze(), labels.squeeze())                      
            valid_epoch_loss += loss.detach()
            
        print(f"Epoch {i}: total VALID loss: {valid_epoch_loss/len(valid_loader)}")
        
        # test the current model 
        """test_epoch_loss = 0
        for (inputs, labels) in test_loader:

            inputs, labels = inputs.to(device), labels.to(device)            
            output = model(inputs, h)
            loss = loss_fn(output.squeeze(), labels.squeeze())                      
            test_epoch_loss += loss.detach()
            
        print(f"Epoch {i}: total TEST loss: {test_epoch_loss/len(test_loader)}")
        print("_______________________________")"""

if __name__ == "__main__":
    

    # generate dummy input for testing 
    #inp = torch.randn(batch_size, seq_len, input_dim)
    #hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
    #cell_state = torch.randn(n_layers, batch_size, hidden_dim)
    #hidden = (hidden_state, cell_state)
    #(out, hidden) = lstm_layer(inp, hidden)
    
    train_data, val_data, test_data = load_data('real_for_all_podaci.csv') 
    
    input_dim = train_data.shape[1]

    train_dataset = Dataset(train_data, seq_len)
    
    val_dataset = Dataset(val_data, seq_len)
    test_dataset = Dataset(test_data, seq_len)


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    
         
    model = Net(input_dim,  drop_prob = 0)
    model.to(device)
    #loss_fn = nn.MSELoss(reduction='sum') # squared error loss
    loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    
    
    
    train(model, train_loader, val_loader, test_loader, loss_fn, optimizer)




















