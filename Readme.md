# Energy Efficient Reinforcement Learning (EERL)



## Introduction

This project is the implementation of the algorithms in the following technique notes:

> [Energy-Efficient Reinforcement Learning](https://leonardo-h.github.io/JiaweiHuang/Docs/EERL.pdf)
> Jiawei Huang, 2018.

We aim at training Binary Neural Networks for policy evaluatioin. Please refer to the notes for details.



## CopyRight

We built our code based on OpenAI baseline. Specifically, each `.py` file in this project are modified from the original file in OpenAI baseline.



## Installation

Please first download OpenAI baselines (https://github.com/openai/baselines).

Then checkout to the following history commit, and follow the instructions to install OpenAI baselines.

```
git checkout 1f3c3e33e7891cb350562757f29cb32ec647efd0
```

Finally, download and copy this project to the path `./baselines/baselines`.



## Experiments

We presented four algorithms in with two different learning strategy.

### Imitation Learning

To run code of algorithm 1 & 2 in this section. please first train an "experts" network (via DQN, for example) and configure the `load_model_path` in `imitation_train.py` and `svgd_imitation_train.py`.

#### Algorithm 1: Imitating an Expert by DAgger (BIL)

```shell
python imitation_train.py --env your_env --seed your_seed --en your_bnn_number
```



#### Algorithm 2: Imitating an Expery by DAgger with SVGD Update Method (BIL + SVGD)

```shell
python svgd_imitation_train.py --env your_env --seed your_seed --en your_bnn_number
```





### Bootstrapping Q Learning

#### Algorithm 3: Learning from Reward (BLR)

```shell
python svgd_imitation_train.py --env your_env --seed your_seed --en your_bnn_number
```



#### Algorithm 4: Learning from Reward with SVGD Update Method (BLR + SVGD)

```shell
python svgd_imitation_train.py --env your_env --seed your_seed --en your_bnn_number
```





