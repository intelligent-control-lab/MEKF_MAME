# MEKF_MAME
**M**odified **E**xtended **K**alman **F**ilter with generalized exponential **M**oving **A**verage and dynamic **M**ulti-**E**poch update strategy (MEKF_MAME)

Pytorch implementation source coder for paper "Robust Nonlinear Adaptation Algorithms for Multi-TaskPrediction Networks".

**We will release the code after paper acceptance soon**

In the paper, EKF based adaptation algorithm  MEKF_λ was introduced as an effective base algorithm for online adaptation. In order to improve the convergence property of MEKF_λ, generalized exponential moving average filtering was investigated. Then this paper introduced a dynamic multi-epoch update strategy, which can be compatible with  any optimizers. By combining all extensions with base MEKF_λ algorithm, robust online adaptation algorithm MEKF_MAME was created.

