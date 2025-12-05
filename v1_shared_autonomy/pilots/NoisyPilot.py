import numpy as np
from stable_baselines3 import PPO
from pilots.pilot import Pilot

class NoisyPilot(Pilot):
    """
    A specialized Pilot that introduces random noise into the action selection process.
    This class allows a portion of the actions to be chosen randomly, adding stochasticity to the policy.
    """

    def __init__(self, model, seed=1, alpha=0.5, action_space=6):
        """
        Initializes the NoisyPilot object.

        Args:
            model (PPO): A trained PPO reinforcement learning model.
            seed (int, optional): An integer used to initialize the random number generator. Defaults to 1.
            alpha (float, optional): Coefficient for noise probability. If alpha=0.3, 30% of the time a random action is selected. Defaults to 0.5.
            action_space (int, optional): The number of discrete actions the agent can perform. Defaults to 7.
        """
        super().__init__(model, seed, alpha, action_space)


    def choose_action(self, obs):
        #TODO check type for obs
        """
        Implements the NoisyPilot policy, which selects actions either from the model or randomly based on the noise coefficient.

        Args:
            obs (array-like): An observation returned by the environment (Gym.Env).

        Returns:
            tuple:
                - action (int): The selected action (either random or predicted by the model).
                - _states (list): Additional state information (if applicable, returned when using the model).
        """
        action = None
        _states = []

        if self._rng.random() <= self._alpha:
            action = self._rng.integers(0, self._action_space)
        else:
            action, _states = self.get_action_from_model(obs)

        return action, _states
