import torch
from torch.utils.data import DataLoader
from train_test_split import load_data
from data_set import Dataset
from data_set import Dataset_smsd 
from data_set import  weighted_mse_loss
#import numpy as np
from model import Net

# ugly to have it as globals, but don't care for now...

batch_size = 1
nr_sim = 30 # number of simulated sets

seq_len = 3 # input sequence lenght
epochs = 50
lr_rate=0.001
hidd_dim = 512
#hidd_dim2 = 1024
att = True
nr_days = 2 # number of forcasting days
dataFile = f'sim-{nr_sim}-PRAM-real_for_all_podaci_novo.csv'

use_weights = False # MSE loss function either uses weights or it does not
use_model = 'multiple_models_sim_data' # ['single_model_orig_data', 'multiple_models_sim_data', 'single_model_sim_data']

if use_model == 'single_model_orig_data':
    nr_sim = 0 # 0 indicated that the original (and not simulated) data should be used
    
if use_model == 'single_model_sim_data':
    batch_size = 1 # must be one, since batch dimension in input tenzor will be used for multiple simulated inputs
    
    

def train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, scheduler, device, target, saveModelPath):


    bestLoss = 1e9
    
    
    for i in range(epochs):
        train_epoch_loss = 0

        model.train()
        #h = model.init_hidden(batch_size, device)
        for (inputs, meteo, labels, weights) in train_loader:
            inputs, meteo, labels, weights= inputs.to(device), meteo.to(device), labels.to(device), weights.to(device)
            model.zero_grad()
            import pdb
            pdb.set_trace()
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
            torch.save(model.state_dict(), saveModelPath)
            bestLoss = valid_epoch_loss/len(valid_loader)
        
        test_epoch_loss = 0
        for (inputs, meteo, labels, weights) in test_loader:
            inputs, meteo, labels, weights = inputs.to(device), meteo.to(device), labels.to(device), weights.to(device) 
            output = model(inputs, meteo)#, h)
            loss = loss_fn(output.squeeze(), labels.squeeze(), weights.squeeze())                      

            test_epoch_loss += loss.detach()
        print(f"Epoch {i}: total TEST loss: {test_epoch_loss/len(test_loader)}")
    


if __name__ == "__main__":
    
    #TARGET = 'PRBR'
    TARGET = 'PRAM'
    #TARGET = 'PRTR'

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    train_data, val_data, test_data = load_data(dataFile, preproc='lognormalize', target=TARGET, nr_sim = nr_sim, use_weights=use_weights)
         
    #loss_fn = torch.nn.MSELoss(reduction='mean') # squared error loss
    loss_fn = weighted_mse_loss(reduction='mean') 
    
    val_dataset = Dataset(val_data, seq_len, nr_days, TARGET)
    test_dataset = Dataset(test_data, seq_len, nr_days, TARGET)    
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    if use_model == 'single_model_orig_data':
        train_dataset = Dataset(train_data, seq_len, nr_days, TARGET)
        #val_dataset = Dataset(val_data, seq_len, nr_days, TARGET)
        #test_dataset = Dataset(test_data, seq_len, nr_days, TARGET)
    
        input_dim = train_dataset[0][0].shape[1]
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        #val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        #test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
        model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len, attention = att, device = device)
    
        model.to(device)
   
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.001)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, weight_decay=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-6)
        savePath = f'models/{TARGET}/batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}_useWeights_{use_weights}_best_meteo.weights'
        train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device, target=TARGET, saveModelPath = savePath)
    
    elif use_model == 'multiple_models_sim_data': # train nr_sim different models

        
        for i in range(nr_sim):
            train_dataset = Dataset(train_data, seq_len, nr_days, TARGET, nr_sim = f'{i}-sim')            

            input_dim = train_dataset[0][0].shape[1]
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
         
            model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len, attention = att, device = device) 
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-6)
            savePath = f'models/{TARGET}/simModels/{i}-sim/batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}_useWeights_{use_weights}_best_meteo.weights'
            train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device, target=f'{i}-sim', saveModelPath = savePath)
            
    elif use_model == 'single_model_sim_data': # train single model on multiple simulated datasets
        pass
                
    else:
        raise Exception("use_model should be set to either of ['single_model_orig_data', 'multiple_models_sim_data', 'single_model_sim_data']")
    
    