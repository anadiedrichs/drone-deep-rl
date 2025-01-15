import numpy as np
from stable_baselines3 import PPO
from pilots.pilot import Pilot

class LaggyPilot(Pilot):
    """
    A specialized Pilot that simulates laggy behavior by occasionally repeating the last action.
    This introduces temporal dependency in the policy, mimicking a delayed response system.
    """

    def __init__(self, model, seed=1, alpha=0.5, action_space=7):
        """
        Initializes the LaggyPilot object.

        Args:
            model (PPO): A trained PPO reinforcement learning model.
            seed (int, optional): An integer used to initialize the random number generator. Defaults to 1.
            alpha (float, optional): The lag probability. For example, if alpha=0.3, the pilot behaves laggy 30% of the time. Defaults to 0.5.
            action_space (int, optional): The number of discrete actions the agent can perform. Defaults to 7.
        """
        super().__init__(model, seed, alpha, action_space)

        self.last_action = None

    def choose_action(self, obs):
        #TODO check type for obs
        """
        Implements the LaggyPilot policy, which either selects a new action or repeats the last action based on the lag probability.

        Args:
            obs (array-like): An observation returned by the environment (Gym.Env).

        Returns:
            tuple:
                - action (int): The selected action. If lagging, this will be the previously executed action.
                - _states (list): Additional state information (if applicable, returned when using the model).
        """
        _states = []

        if self._rng.random() <= self._alpha or self.last_action is None:
            # Update last_action with a new action from the model
            action, _states = self.get_action_from_model(obs)
            self.last_action = action

        # Otherwise, return the previously executed action
        return self.last_action, _states
