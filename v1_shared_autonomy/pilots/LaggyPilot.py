import numpy as np
from stable_baselines3 import PPO


class LaggyPilot:

    def __init__(self, model: PPO, seed=1):
        """
        LaggyPilot constructor

        Args:
            seed (int): an integer number to initialize the random generator
            model (PPO): a trained PPO reinforcement learning model

        Returns:
            tipo: LaggyPilot
        """
        if model is None or not isinstance(model, PPO):
            raise ValueError("The model must be a non-null instance of PPO.")

        self.seed = seed
        np.random.seed(self.seed)
        self.model = model
        self.last_action = None

    def choose_action(self, obs, lag_prob=0.3):
        """
        This method implements the LaggyPilot policy.

        Args:
            obs: an observation returned by an environment (Gym.Env)
            lag_prob: If lag_prob is 0.3, a 30% of the time will behave as a laggy
            pilot, otherwise it will return the action of the normal policy.

        Returns:
            tipo: action (int)
        """
        action, _states = self.model.predict(obs)
        if self.last_action is None or np.random.random() <= lag_prob:
            self.last_action = action
        return self.last_action
