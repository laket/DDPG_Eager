This repository is an example code to compare Tensorflow Eager Execution with PyTorch.
 You need OpenAI gym with MuJoCo to run these codes.

The codes in this repository was modified from https://github.com/sfujim/TD3.
The original code was written for the paper,

```
Scott Fujimoto, Herke van Hoof, David Meger,
Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018
```

If you want to use this code, confirm the conditions of use in https://github.com/sfujim/TD3.


If you are Japanese, see the qiita article which explains these codes.

# Usage

```
# Pytorch Mode
python3 main.py --policy_name DDPG
# Eager Mode
python3 main.py --policy_name tf_DDPG
# Eager Mode + defun
python3 main.py --policy_name tf_DDPG_fast

```
