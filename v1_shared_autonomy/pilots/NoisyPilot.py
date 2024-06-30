import numpy as np
from stable_baselines3 import PPO


class NoisyPilot:

    action_space: int

    def __init__(self, model: PPO, seed=1, action_space=7):
        """
        NoisyPilot constructor

        Args:
            seed (int): an integer number to initialize the random generator
            model (PPO): a trained PPO reinforcement learning model
            action_space: how many discrete actions can the agent perform

        Returns:
            tipo: NoisyPilot
        """
        if model is None or not isinstance(model, PPO):
            raise ValueError("The model must be a non-null instance of PPO.")

        self.seed = seed
        np.random.seed(self.seed)
        self.model = model
        self.action_space = action_space

    def choose_action(self, obs, noise_prob=0.3):
        """
        This method implements the LaggyPilot policy.

        Args:
            obs: an observation returned by an environment (Gym.Env)
            noise_prob: If noise_prob is 0.3, a 30% of the time will select a
            random action, otherwise it will return the action of the normal policy.

        Returns:
            tipo: action (int)
        """
        action = None

        if np.random.random() > noise_prob:
            
            action, _states = self.model.predict(obs)

        else:
            action = np.random.randint(0, self.action_space)

        return action
