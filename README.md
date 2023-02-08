"# polen_forcasting" 

# Pollen forecasting

This repository contains code for training LSTM network whose main goal is to forecast day-by-day pollen concentration.
In provided data there is three types of pollen: ragweed (PRAM), grass pollen (PRTR) and birch (PRBR).
This repository is supplementary material for ... <!--add paper title -->

## Running the code
Before running the code, you should change parameters in main.py, mainly: batch size, number of simulated datasets (here it is always 30), input sequance length, number of epochs for training, learning rate for training, hidden dimension for model, wheter or not attention is used, dropout rate, number of days for which you want to predict pollen concentration, name of file where data is stored, test, validation and excluded years, wheter or not you want to use weights in loss function and finally, which model you want to use.
Additionaly, you should make folders: models > {target} > {model type}, e.g. models>PRAM>SMOD where trained model will be saved. After that you simply run the training process by running: 
```
python main.py
```

## Evaluating the model
After training the model, you can evaluate it using iPython notebooks evaluate_model.ipynb (for single model with original data and single model with simulated data models) and evaluate_multipleModels.ipynb (for multiple models with simulated data). Again, you shoud change parameters and path file to match trained model.

In those notebook you can generate graphs showing real and predicted values, such as:
<p align="center"><img src="https://github.com/dmatijev/polen_forcasting/blob/main/images/real_predicted_2020_2021_SMSD.jpg?raw=true" width="350" ></p>

<!--Koliko u sirinu ici? Trebam li napisati da imamo 3 modela, da imamo attention? Treba li pisati i sto je u kojem fileu? npr. da se pomocu simulate data simuliraju podaci za SMSD i MMSD? -->
