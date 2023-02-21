# Pollen forecasting using PollenNet

This repository contains code for training RNN network whose main goal is to forecast day-by-day pollen concentration.
This repository is supplementary material for the paper PollenNet -  a deep learning approach for the prediction of airborne pollen concentrations.

## Running the code
Before running the code, you should change parameters in main.py, mainly: batch size, number of simulated datasets (here it is always 30), input sequence length, number of epochs for training, learning rate for training, hidden dimension for model, dropout rate, number of days for which you want to predict pollen concentration, name of file where data is stored, test, validation and excluded years, and finally, which model you want to use.
Additionaly, you should make folders: models > {target} > {model type}, e.g. models>PRAM>SMOD where trained model will be saved. After that you simply run the training process by running: 
```
python main.py
```

## Evaluating the model
After training the model, you can evaluate it using iPython notebooks evaluate_model.ipynb (for single model with original data (PollenNet in the paper) and single model with simulated data models (SM in the paper)) and evaluate_multipleModels.ipynb (for multiple models with simulated data (RM in the paper)). Again, you shoud change parameters and path file to match the trained model.

In those notebooks you can generate graphs showing real and predicted values, such as:
<p align="center"><img src="https://github.com/dmatijev/polen_forcasting/blob/main/images/real_predicted_2020_2021_SMSD.jpg?raw=true" width="350" ></p>

<!--Koliko u sirinu ici? Trebam li napisati da imamo 3 modela, da imamo attention? Treba li pisati i sto je u kojem fileu? npr. da se pomocu simulate data simuliraju podaci za SMSD i MMSD? -->
