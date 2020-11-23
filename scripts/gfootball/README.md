# Training agent for GFootball
Training and RL agent to play [GFootball](https://github.com/google-research/football).

The GFootball environment presents a number of challenges that make it difficult for a agent to learn. Significantly, adequate exploration of a massively dimensional space with very infrequent rewards.

This set of scripts attempts to overcome some of these hurdles using the following approaches:
  - Pretraining - using data from historical games played by other agents crafted by in a [Kaggle competition](https://www.kaggle.com/c/google-football).
  - Exploration - using the actions proposed by a hand crafted bot, rather than random actions with EpsilonGreedy
  
Overall this approach can train an agent that ~1000 points in the [Kaggle competition](https://www.kaggle.com/c/google-football), with the majority of the performance gained by the pretraining. This is competitive with other agents in the competition, and somewhat competitive against the built in environment AI. 
Improvements in performance from this point are likely to come from the following, in no particular order.
  - Implementing more improvements to the DQN (eg. prioritized replay buffer) or switching to a more advanced RL technique entirely.
  - Engineering other environment features, such as those used by various bots in the Kaggle competition.
  - More compute. I've only run the RL training on CPU so far (due to running locally in a Linux VM), and it's likely to need a LOT of training. GPU training would allow for:
     - Switching to a more complex representation for the environment observations - work so far has focused on using the Simple115 features, rather than Raw/SMM versions.
     - Switching to a more complex NN models.
  - More advanced exploration strategies, for example simultaneous experience collection from multiple policies/sources (eg. multiple other bot implementations playing independently) rather randomly taking individual actions from a single bot.


# Setup 
0) Install GFootball
    ```bash
    pip install gfootball
    ```
    See also [GFootball repo](https://github.com/google-research/football) for more detailed instructions.

    In case of problems, two additional methods to try are available [here](https://github.com/garethjns/kaggle-football/tree/main/setup) 

1) Install other requirements
    https://github.com/garethjns/kaggle-football - includes code for downloading games, and other various useful things. 
    
    ```bash 
    pip install reinforcement_learning_keras
    git clone https://github.com/garethjns/kaggle-football.git
    pip install -e kaggle-football
    ```

# Running
There are 5 steps to run:

1) Download data and generate features for pretraining
2) Pretrain a classification model
3) Train an RL model using pretrained weights and extra experience from a bot
4) Compare between the pretrained model, rl model, and the bot.
5) Play the rl model against the built in AI (and the classification model for comparison.)

Run the scripts in order, by default they're set up to download a very small amount of data from the Kaggle AI and to run very short training. So don't expect good performance without tweaking!

## 1) Download data for pretraining
Download historical games from the Kaggle API and create a dataset.

Games are downloaded as json including the raw observations at each step. These are zipped and saved.

After downloading the games are collected into a .hdf file. The features are generated from the json to create
structured data of the simple, raw, and smm obs. SMM is turned off for now.

```bash
python3 -m scripts.gfootball.1_download_data
```

## 2) Pretrain classification model
This script trains a classification model, the weights of which will be used later to create the RL agents model.

It loads from the data downloaded and structured by 1_download_data.py and casts the problem as a classification
problem. ie. Given these features (the observation features, s115, smm, raw, whatever), what action will the "bot" take?
"Bot" here being the agents submitted to the Kaggle competition; they could be hardcoded bots or RL agents, we don't
care for now.

Training is done using the s115 features and optionally can add the raw features. Adding the raw features causes massive
overfiting and is off for now.

The RL agent will use a frame buffer wrapper which stacks the last and current observation. This is handled here by
offsetting the rows of the dataset and passing the last and current features as input (first steps are discarded).
The input shape when using the s115 features is therefore 115 * 2, where the columns are arranged as:
[s115 from the last observation, s115 from the current observation].

```bash
python3 -m scripts.gfootball.2_pretrain_classification_model
```

## 3) Train reinforcement learning model
This script copies the weights from the classification model trained in the previous step to a DQN agent and begins RL training.

It uses the [open rules bot](https://www.kaggle.com/mlconsult/best-open-rules-bot-score-1020-7) (see rlk.environments.gfootball.bots.open_rules_bot) as an additional policy for experience
collection. Note that this is done in a similar way to EpsilonGreedy (rather than the bot playing on its own). On each
step, depending on epsilon, the agent will either sample an action from its own model, or from the bot
(rather than a totally random action)

Communication between the rlk env and the bot (which uses the Kaggle completion api and expects raw observations rather
than, eg. simple115 from the gym-wrapped env, is handled using the SimpleAndRawObsWrapper. This returns the simple115
observations to rl agent as normal, but additionally dumps the raw observations to disk for the bot to use.

There's also a frame buffer wrapper that stacks 2 frames.

```bash
python3 -m scripts.gfootball.3_train_rl_model
```

## 4) Compare between the pretrained model, rl model, and the bot
This script compares the pretrained model, rl model, and the open rules bot used during RL training.

```bash
python3 -m scripts.gfootball.4_compare_models_and_bot
```

## 5) Play the rl model against the built in AI (and the classification model for comparison)
This script evaluates the rl agent (and also pretrained agent for comparison) against the built in AI in the GFootball
environment.

```bash
python3 -m scripts.gfootball.5_eval_model_against_ai
```

# Submit agent
kaggle_agent_classification_model.py and kaggle_agent_rl_model.py are compatible with Kaggle environments and can be submitted to the [Google Research Football with Manchester City F.C.](https://www.kaggle.com/c/google-football) competition.


```bash
cp scripts/gfootball/kaggle_agent_classification_model.py main.py
tar -czvf submission.tar.gz main.py nn_s115_pretrained_model
```

or

```bash
cp scripts/gfootball/kaggle_agent_rl_model.py main.py
tar -czvf submission.tar.gz main.py nn_s115_rl_model
```
