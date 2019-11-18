# Energy Efficient Reinforcement Learning

## Introduction

This project aims at training Binary Neural Networks for policy evaluatioin. We plan to submit this work to IJCAI 2019 and open source code after submission.



## Algorithms

There are four algorithms in this project:

### Imitation Learning Part

##### 1. Imitating an expert by DAgger.

imitation_train.py + ./graph/imitation_graph.py

##### 2. Imitating an expery by DAgger with SVGD update method.

svgd_imitation_train.py +./graph/svgd_imitation_graph.py

#### Bootstrapping Q Learning Part

##### 3. Learning from reward.

advantage_learning.py + ./graph/advantage_learning_graph.py

##### 4. Learning from reward with SVGD update method.

svgd_advantage_learning.py + ./graph/svgd_advantage_learning_graph.py



Besides, BNN models defined in model.py.