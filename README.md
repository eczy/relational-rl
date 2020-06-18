# relational-rl

This repo contains 4 models:
* Q-learning
* SARSA
* Deep Q-learning
* Relational Q-learning

Below are the instructions:
1. Open the "relational-rl" directory from terminal

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the following commands to run each of the models:
* Q-learning
```python
python main_TabularQLearning.py
```

* SARSA
```python
python main_TabularSarsa.py
```

* Deep Q-learning
```python
python main.py baseline
```

* Relational Q-learning
```python
python main.py relational
```

# Acknowledgement:
`box_world_env.py` and `boxworld_gen.py` are from Nathan Grinsztajn's BoxWorld implementation found at `https://github.com/nathangrinsztajn/Box-World`.

