"""
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

"""

from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from kaggle_football.api.data.hdf_repository import HDFRepository
from kaggle_football.models.callbacks import Callbacks
from kaggle_football.models.time_dimension import TimeDimension
from tensorflow.keras.backend import one_hot

from rlk.agents.components.helpers.virtual_gpu import VirtualGPU
from rlk.agents.models.denser_nn import DenserNN


def training_summary(mod, x_train, x_test, y_train, y_test):
    train_preds = mod.predict(x_train).argmax(axis=1)
    test_preds = mod.predict(x_test).argmax(axis=1)

    random_baseline = 1 / 19
    print(f"Train score: {np.round(np.mean(train_preds == y_train), 2)} "
          f"vs {np.round(random_baseline, 2)}")
    print(f"Test score: {np.round(np.mean(test_preds == y_test), 2)} "
          f"vs {np.round(random_baseline, 2)}")


def various_plots(mod, x_train: Union[List[np.ndarray], np.ndarray], x_test: Union[List[np.ndarray], np.ndarray],
                  y_train: np.ndarray, y_test: np.ndarray) -> None:
    """This can be slow with large n."""

    plot_data_train = pd.DataFrame({'preds': mod.predict(x_train).argmax(axis=1),
                                    'true': y_train, 'train': 1})
    plot_data_test = pd.DataFrame({'preds': mod.predict(x_test).argmax(axis=1),
                                   'true': y_test, 'train': 0})
    plot_data = pd.concat((plot_data_train, plot_data_test), axis=0)

    fig, axs = plt.subplots(nrows=2)
    sns.displot(ax=axs[0], data=plot_data, x='true', hue='train', kde=False, bins=19)
    plt.show()
    sns.displot(ax=axs[1], data=plot_data, x='preds', hue='train', kde=False, bins=19)
    plt.show()

    sns.jointplot(data=plot_data_train, y='true', x='preds', kind='kde', title='Train set')
    plt.show()
    sns.jointplot(data=plot_data_test, y='true', x='preds', kind="kde", title='Test set')
    plt.show()

    sns.displot(y_train, kde=False, bins=19)
    plt.show()

    plt.plot(mod.predict(x_train[0, :].reshape(1, -1)).squeeze())
    plt.show()


def train_nn_s115_raw(use_raw: bool = False, roll_steps: int = 1):
    """

    :param use_raw: Bool indicating whether or not to add additional features based on the raw observations.
    :param roll_steps: Number of steps to offset the dataset by to obtain past observations. Should match number of
                       frames the RL agent will use in its frame buffer. Currently can b 0 (use just current
                       observation) or 1 (use last and current).
    """

    VirtualGPU(1500)

    repo = HDFRepository().set_path(f"downloaded_games")

    train_episodes, test_episodes = repo.split(train_prop=0.95)
    actions, s115, _, raw = repo.load_episodes(keys=[repo.actions_key, repo.s115_key, repo.raw_key])

    if use_raw:
        x = np.concatenate([s115, raw], axis=1)
    else:
        x = s115

    if roll_steps > 0:
        # Check all episodes have 3000 steps
        # This relies on iding episodes with agent, episode, and score. Any clash would be overwritten in the json repo,
        # so any combo should have 3000 steps max.
        semi_unique_ids = (actions['agent_id'].astype(str) +
                           '_' + actions['episode_id'].astype(str) +
                           '_' + actions["updated_score"].astype(str))
        n_steps = np.unique(semi_unique_ids.value_counts())
        n_episodes = len(np.unique(semi_unique_ids))
        assert n_steps == 3000

        td = TimeDimension(n_episode_steps=int(n_steps), n_roll_steps=1).fit(x.shape[0])
        x = td.transform_1d(x)
        actions = actions.reset_index(drop=True).drop(td.idx_to_drop_, axis=0)

        assert x.shape[0] < s115.shape[0]
        assert x.shape[1] > s115.shape[1]
        assert x.shape[0] == actions.shape[0]
        assert x.shape[0] == (s115.shape[0] - n_episodes)
        assert np.unique((actions['agent_id'].astype(str) +
                          '_' + actions['episode_id'].astype(str) +
                          '_' + actions["updated_score"].astype(str)).value_counts()) == 2999

    train_idx = actions['episode_id'].isin(train_episodes).values
    test_idx = actions['episode_id'].isin(test_episodes).values
    x_train = x[train_idx, ...]
    x_test = x[test_idx, ...]
    y_train = actions.iloc[train_idx, :]
    y_test = actions.iloc[test_idx, :]

    print(f"Training with {len(train_episodes)} episodes, totalling {y_train.shape[0]}, rows")
    print(f"Evaluating with {len(test_episodes)} episodes, totalling {y_test.shape[0]}, rows")

    model_class = DenserNN
    mod_arc = model_class(observation_shape=(x_train.shape[1],), n_actions=19,
                          learning_rate=0.001, output_activation='softmax', dueling=False)
    mod = mod_arc.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    mod.fit(x_train, one_hot(y_train[repo.target], num_classes=19), callbacks=[Callbacks.es, Callbacks.tb],
            validation_split=0.2, epochs=1000, batch_size=5000)
    mod.save(f"nn_s115_pretrained_model")

    training_summary(mod, x_train, x_test, y_train[repo.target], y_test[repo.target])
    various_plots(mod, x_train, x_test, y_train[repo.target], y_test[repo.target])

    return mod


if __name__ == "__main__":
    mod = train_nn_s115_raw()
