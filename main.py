import torch
from torch.utils.data import DataLoader
from train_test_split import load_data
from data_set import Dataset
from data_set import Dataset_SMSD 
from data_set import  weighted_mse_loss
from model import Net
import tqdm

# ugly to have it as globals, but don't care for now...

#TARGET = 'PRBR' #birch pollen
TARGET = 'PRAM' #ragweed pollen
#TARGET = 'PRTR' #grass pollen

#parameters:
batch_size = 32
nr_sim = 30 # number of simulated sets
seq_len = 7 # input sequence lenght
epochs = 50
lr_rate=0.001
hidd_dim = 256
att = True 
dropout = 0.1
nr_days = 2 # number of forcasting days
dataFile = f'sim-{nr_sim}-{TARGET}-jump_weights-real_for_all_NS_2000_2021.csv'
test_years = [2020,2021] #years from dataset used for testing
valid_years = [2018,2019] #years from dataset used for validation
exclude_years = [] #list of additional years that should be excluded from training (test and valid are already excluded)

use_weights = False # MSE loss function either uses weights or it does not
use_model = 'multiple_models_sim_data' # ['single_model_orig_data', 'multiple_models_sim_data', 'single_model_sim_data']

if use_model == 'single_model_orig_data':
    nr_sim = 0 # 0 indicated that the original (and not simulated) data should be used
    
if use_model == 'single_model_sim_data':
    batch_size = 1 # must be one, since batch dimension in input tenzor will be used for multiple simulated inputs
    
    

def train(model, train_loader, valid_loader, test_loader, loss_fn, optimizer, scheduler, device,  saveModelPath):
    bestLoss = 1e9
    
    for i in range(epochs):
        train_epoch_loss = 0

        model.train()
        for (inputs, meteo, labels, weights) in tqdm.tqdm(train_loader):
            
            inputs, meteo, labels, weights= inputs.to(device), meteo.to(device), labels.to(device), weights.to(device)
            model.zero_grad()
            
            if use_model == 'single_model_sim_data':
                inputs = torch.squeeze(inputs, dim=0)
                shared_weights = True
            else:
                shared_weights = False

            output = model(inputs, meteo, shared_weights = shared_weights)
            
            loss = loss_fn(output.squeeze(), labels.squeeze(), weights.squeeze())

            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach()
        print(f"Epoch {i}: total TRAIN loss: {train_epoch_loss/len(train_loader)}")

        scheduler.step(train_epoch_loss)
        
        # validate the current model (this should get encapsulated to its own function)
        model.eval()
        valid_epoch_loss = 0
        for (inputs, meteo, labels, weights) in valid_loader:
            inputs, meteo, labels, weights = inputs.to(device), meteo.to(device), labels.to(device), weights.to(device) 
            output = model(inputs, meteo)
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
            output = model(inputs, meteo)
            loss = loss_fn(output.squeeze(), labels.squeeze(), weights.squeeze())                      

            test_epoch_loss += loss.detach()
        print(f"Epoch {i}: total TEST loss: {test_epoch_loss/len(test_loader)}")
    


if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    train_data, val_data, test_data = load_data(dataFile, test_years=test_years, valid_years=valid_years, exclude_years=exclude_years, preproc='lognormalize', target=TARGET, nr_sim = nr_sim, use_weights=use_weights)
         
    loss_fn = weighted_mse_loss(reduction='mean') 
    
    val_dataset = Dataset(val_data, seq_len, nr_days, TARGET)
    test_dataset = Dataset(test_data, seq_len, nr_days, TARGET)    
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    if use_model == 'single_model_orig_data':
        print(f"Running {use_model} with attention = {att}...")
        train_dataset = Dataset(train_data, seq_len, nr_days, TARGET)
    
        input_dim = train_dataset[0][0].shape[1]
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    
        model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len,  dropout = dropout, attention = att, device = device)
    
        model.to(device)
   
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-3)
        savePath = f'models/{TARGET}/SMOD/batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}-use_weights_{use_weights}-dropout_{dropout}-best.weights'
        train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device, saveModelPath = savePath)
    
    elif use_model == 'multiple_models_sim_data': # train nr_sim different models
        print(f"Running {use_model} with attention = {att}...")
        
        for i in range(nr_sim):
            train_dataset = Dataset(train_data, seq_len, nr_days, TARGET, sim_label = f'{i}-sim')            

            input_dim = train_dataset[0][0].shape[1]
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
         
            model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len,  dropout = dropout, attention = att, device = device) 
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-3)
            savePath = f'models/{TARGET}/MMSD/{i}-sim_batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}-use_weights_{use_weights}-dropout_{dropout}-best.weights'
            train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device, saveModelPath = savePath)
            
    elif use_model == 'single_model_sim_data': # train single model on multiple simulated datasets
        print(f"Running {use_model} with attention = {att}...")
        train_dataset = Dataset_SMSD(train_data, seq_len, nr_days, TARGET, nr_sim = nr_sim)
        input_dim = train_dataset[0][0].shape[2]
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
         
        model = Net(input_dim,  hidden_dim = hidd_dim, nr_days = nr_days, seq_len = seq_len, dropout = dropout, attention = att, device = device) 
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-3)
        savePath = f'models/{TARGET}/SMSD/batch_size_{batch_size}-seq_len_{seq_len}-nr_days_{nr_days}-lr_{lr_rate}-hidd_dim_{hidd_dim}-att_{att}-use_weights_{use_weights}-dropout_{dropout}-best.weights'
        
        train(model, train_loader, val_loader, test_loader, loss_fn, optimizer, scheduler, device=device,  saveModelPath = savePath)
                
    else:
        raise Exception("use_model should be set to either of ['single_model_orig_data', 'multiple_models_sim_data', 'single_model_sim_data']")
    
    