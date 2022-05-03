# Reinforcement Learning Project

# Members:
- Andre
- Warren
- Oguz
- Lawrence
- Tom
- Sam

# Instructions For Usage:

Firstly, you will need to download the needed packages to run the scripts.

We're assuming you have created a new environment to download these dependencies.
If you already have an RL environment, it should probably work, as the packages we have used are very basic.

To do this, in the root directory of this folder, run:
```
pip install -r requirements.txt
```

Now, all the dependencies should be installed.
Sometimes ALE will have licensing issues, so you might need to run:
```
pip install gym[atari,accept-rom-license]==0.21.0
```

Additionally, depending on your GPU, a different install of CUDA might be needed - for example, if you're running this on the RTX3090, you would need:
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Hopefully all your dependencies should be installed, and your environment activated.

Now, we briefly mention how to:
  1st: Train your agent.
  2nd: Test the result performance of your agent.

# Training your agent
To select which add-ons or DQN variation you would like to run, you must change these settings in config.json.
You can also adjust the hyperparameters (epsilon, total episode count, gamma, etc.) in here. Only thing worth noting is that the True and False values must have a capitalised T, and F. 

When you have adjusted the hyperparameters and selected the DQN variations to your liking in config.json, save it.

Now, you can train your agent.

To run the training script, run the command:
```
python src/main.py
```
Note that if you're on Linux, you might get an error because of how we save our files.
This is to due with the backslashes on line 65 in main.py.
If the script is creashing because of this, change the value of path_to_file to:
```
path_to_file = f"(pwd)/score_logs/boxing-v5_{method}DQN{custom_name}.txt"
```
where pwd would be the output you get from running `pwd` in the root directory of our folder.




