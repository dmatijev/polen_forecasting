import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_test_split import load_data
from data_set import Dataset
from data_set import  weighted_mse_loss
#import numpy as np
from model import Net

# ugly to have it as globals, but don't care for now...

batch_size = 32 

seq_len = 3 # input sequence lenght
epochs = 500
lr_rate=0.005
hidd_dim = 512
#hidd_dim2 = 1024
att = True
nr_days = 2 # number of forcasting days

use_weights = False # MSE loss function either uses weights or it does not
use_model = 'single_model_orig_data' # ['single_model_orig_data', 'multiple_models_sim_data', 'single_model_sim_data']

if use_model == 'single_model_sim_data':
    batch_size = 1 # since batch dimension in input tenzor will be used for multiple simulated inputs
    

def train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, scheduler, device, target):


    bestLoss = 1e9
    
    
    for i in range(epochs):
        train_epoch_loss = 0

        model.train()
        #h = model.init_hidden(batch_size, device)
        for (inputs, meteo, labels, weights) in train_loader:
            inputs, meteo, labels, weights= inputs.to(device), meteo.to(device), labels.to(device), weights.to(device)

            model.zero_grad()

            
            output = model(inputs, meteo)#, h)


            loss = loss_fn(output.squeeze(), labels.squeeze(), weights.squeeze())

            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_epoch_loss += loss.detach()
        print(f"Epoch {i}: total TRAIN loss: {train_epoch_loss/len(train_loader)}")

        scheduler.step(train_epoch_loss)
        
        # validate the current model (this should get encapsulated to its own function)
        model.eval()
        valid_epoch_loss = 0
        for (inputs, meteo, labels, weights) in valid_loader:
            inputs, meteo, labels, weights = inputs.to(device), meteo.to(device), labels.to(device), weights.to(device)          
            output = model(inputs, meteo)#, h)
            loss = loss_fn(output.squeeze(), labels.squeeze(), weights.squeeze())                      

            valid_epoch_loss += loss.detach()

        print(f"Epoch {i}: total VALID loss: {valid_epoch_loss/len(valid_loader)}")
        if valid_epoch_loss/len(valid_loader) < bestLoss:
            print('best model found, saving...')
            torch.save(model.state_dict(), f'models/{target}/batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}_best_meteo.weights')
            bestLoss = valid_epoch_loss/len(valid_loader)
        
        #torch.save(model.state_dict(), f'models/{target}/epoch_{i}-batch_size_{batch_size}-lr_{lr_rate}-hidd_dim_{model.hidden_dim}.weights')
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
    
    #TARGET = 'PRBR'
    TARGET = 'PRAM'
    #TARGET = 'PRTR'

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # generate dummy input for testing
    #inp = torch.randn(batch_size, seq_len, input_dim)
    #hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
    #cell_state = torch.randn(n_layers, batch_size, hidden_dim)
    #hidden = (hidden_state, cell_state)
    #(out, hidden) = lstm_layer(inp, hidden)

    train_data, val_data, test_data = load_data('sim-10-real_for_all_podaci.csv', preproc='lognormalize', target=TARGET, nr_sim = 0, use_weights=use_weights)
    
    
    

    train_dataset = Dataset(train_data, seq_len, nr_days, TARGET)
    val_dataset = Dataset(val_data, seq_len, nr_days, TARGET)
    test_dataset = Dataset(test_data, seq_len, nr_days, TARGET)

    input_dim = train_dataset[0][0].shape[1]
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
 
    
    #model = Net(input_dim,  hidden_dim = hidd_dim, hidden_dim2 = hidd_dim2, nr_days = nr_days, seq_len = seq_len, attention = att)
    model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len, attention = att, device = device)

    model.to(device)
    
    #loss_fn = nn.MSELoss(reduction='mean') # squared error loss
    loss_fn = weighted_mse_loss(reduction='mean') 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=.3, threshold=1e-4)

    train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device, target=TARGET)
    
    
    
    