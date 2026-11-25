# Dropout Experiments on Function Learning

A console program for studying how **dropout** and **noisy training data** affect a small neural
network. It was developed in joint work between Saber Gholami and Ali Mohammad Saghiri on adaptive
dropout, alongside the hybrid learning automata in
[Adaptive-Dropout-Algorithm](https://github.com/amsaghiri/Adaptive-Dropout-Algorithm).

## What it does

The program generates data for a chosen mathematical function, optionally corrupts part of it with
noise, trains a neural network on it with or without dropout, and reports how well the network learned.
It also includes classification scenarios in which a network must decide which function a set of
points came from.

Functions available: x², sin x, tan x, √x, |cos x|, ⌊x⌋, a piecewise function, (x²+1)/(x²−4),
|ln x|, x³, and log₂ x.

## What is in the repository

| File | What it is |
| --- | --- |
| `MainProgram.py` | The menu-driven entry point. |
| `FiveL_DropOut_Neural_Network.py` | The five-layer network (`Neural_Net`), with optional dropout. |
| `Data_Manager.py` | Data-set generation, noise injection, and data-set treatment. |
| `Classification_Scenario.py` | Function-classification experiments, with and without pre-learning. |

## Run it

```bash
pip install numpy matplotlib scikit-learn tensorflow
python MainProgram.py
```

The program is interactive. From the main menu you can start a new model, load a saved one, change the
settings (noise type and percentage, dropout on or off and its rate), treat a model's data set, or
open the classification experiments. Models and data are saved under a `Models/` folder created in the
working directory.

## Notes

- The code was written for Keras 2 on Windows. It imports `keras.preprocessing.text`, which newer Keras
  releases no longer include, and it clears the screen with the Windows `cls` command. On other setups
  it may need small changes.
- Some comments in the source are in Persian.
