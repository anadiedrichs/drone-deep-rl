class Pilot():

    def __init__(self, model: PPO, seed=1, action_space=7):
        raise NotImplementedError("Please Implement this method")

    def choose_action(self, obs, noise_prob=0.3):
        raise NotImplementedError("Please Implement this method")