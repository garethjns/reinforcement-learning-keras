"""General script to load a trained agent and play an episode with a monitor wrapper."""

from gym import wrappers

from reinforcement_learning_keras.agents.q_learning.deep_q_agent import DeepQAgent


def load_and_play(agent_name: str, env_spec: str, output_dir: str = None) -> None:
    """Load agent, get it's env, wrap, play."""

    agent = DeepQAgent.load(f"{agent_name}_{env_spec}")

    if output_dir is None:
        output_dir = f'{agent.name}_{env_spec}_monitor_dir'

    # Replace agents environment with a wrapped version of the same env, then play episode. Render doesn't have to be
    # True here to save output to disk.
    agent.env_builder.set_env(wrappers.Monitor(agent.env, output_dir, force=True))
    agent.play_episode(training=False, render=True, max_episode_steps=10000)
    agent.env.close()


if __name__ == "__main__":
    load_and_play(agent_name="DeepQAgent", env_spec="VizdoomBasic-v0")

