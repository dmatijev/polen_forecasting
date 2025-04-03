# Pollen forecasting using PollenNet

## Software License
The software in this repository is licensed under the [MIT License](LICENSE).

## Data License
The data in this repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](LICENSE).

Please be sure to provide appropriate attribution if you use the data, and note that the data cannot be used for commercial purposes.

## Citation
If you use the data from this repository in your research, please cite the following paper:

**Rebeka Čorić, Domagoj Matijević, Darija Marković**. "PollenNet - a deep learning approach to predicting airborne pollen concentrations." *Croatian Operational Research Review*, vol. 14, no. 1, 2023, http://dx.doi.org/10.17535/crorr.2023.0001. 

Please make sure to give appropriate credit to the original authors.

## Description

This repository contains code for training the RNN network, whose primary goal is forecasting day-by-day pollen concentration. 
This repository is supplementary material for the paper "PollenNet - a deep learning approach for predicting airborne pollen concentrations".

## Running the code
Before running the code, you should change parameters in main.py, mainly: batch size, number of simulated datasets (here it is always 30), input sequence length, number of epochs for training, learning rate for training, hidden dimension for the model, dropout rate, number of days for which you want to predict pollen concentration, name of the file where data is stored, test, validation and excluded years, and finally, which model you want to use.
Additionally, it would be best if you made folders: models > {target} > {model type}, e.g., models>PRAM>SMOD where the trained model will be saved. After that, you simply run the training process by running:  
```
python main.py
```

## Evaluating the model
After training the model, you can evaluate it using iPython notebooks evaluate_model.ipynb (for the single model with original data (PollenNet in the paper) and single model with simulated data models (SM in the paper)) and evaluate_multipleModels.ipynb (for multiple models with simulated data (RM in the paper)). Again, you should change the parameters and path file to match the trained model.

In those notebooks, you can generate graphs showing real and predicted values, such as:
<p align="center"><img src="https://github.com/dmatijev/polen_forcasting/blob/main/images/real_predicted_2020_2021_SMSD.jpg?raw=true" width="350" ></p>

<!--Koliko u sirinu ici? Trebam li napisati da imamo 3 modela, da imamo attention? Treba li pisati i sto je u kojem fileu? npr. da se pomocu simulate data simuliraju podaci za SMSD i MMSD? -->
