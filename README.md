# MEKF<sub>EMA-DME</sub>
**M**odified **E**xtended **K**alman **F**ilter with generalized **E**xponential **M**oving **A**verage and **D**ynamic **M**ulti-**E**poch update strategy (MEKF<sub>EMA-DME</sub>)

Pytorch implementation source coder for paper [Robust Online Model Adaptation by Extended Kalman Filter with Exponential Moving Average and Dynamic Multi-Epoch Strategy](https://arxiv.org/abs/1912.01790).


In this paper, inspired by Extended Kalman Filter (EKF), a base adaptation algorithm Modified EKF with forgetting
factor (MEKF<sub>λ</sub>) is introduced first. Then using exponential moving average (EMA) methods, this
paper proposes EMA filtering to the base EKF<sub>λ</sub> in order to increase the convergence rate. 
In order to effectively utilize the samples in online adaptation, this paper proposes a dynamic multi-epoch update strategy to discriminate the “hard” samples from “easy” samples, and sets different weights for them.  With all these extensions, this paper proposes a robust online adaptation algorithm: MEKF with Exponential Moving Average and Dynamic Multi-Epoch update strategy (MEKF<sub>EMA-DME</sub>).


### Requirements
* Python 3.6
* pytorch >=1.1.0
* `pip install -r requirements.txt`

### How to use it
1 .  Offline Neural Network Training
```
python train.py
```
2 .  Online Adaptation
```
python adapt.py
```
You can online adapt the offline trained model with several optimizers, including SGD, Adam, MEKF<sub>λ</sub>, MEKF<sub>EMA-DME</sub>. 

