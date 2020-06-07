from agents.pong.q_learning.deep_q_agent import DeepQAgent

agent = DeepQAgent.example(render=True)
agent.save("test_pong_dqn_script.pkl")
