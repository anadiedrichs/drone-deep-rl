import sys
import os
from gymnasium.spaces.box import Box
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.utilities import *
from copilot.CornerEnv import *
from pilots.LaggyPilot import *
from pilots.NoisyPilot import *
from pilots.pilot import *

class CopilotCornerEnv(SimpleCornerEnvRS10, Pilot):
    """
    A reinforcement learning environment that combines a base environment (`SimpleCornerEnvRS10`)
    with a `Pilot` to create a co-piloted environment. The co-pilot blends actions from the pilot
    with its own policy network, balancing between the two based on a specified alpha.
    """

    def __init__(self, model, pilot, seed=1, alpha=0.5, action_space=6):
        """
        Initializes the co-piloted environment.

        Args:
            model (PPO): A trained PPO model to provide policy-based actions.
            pilot (Pilot): The base pilot providing additional actions.
            seed (int, optional): Seed for random number generation. Defaults to 1.
            alpha (float, optional): Balancing factor between the pilot and co-pilot actions. Defaults to 0.5.
            action_space (int, optional): Number of discrete actions available. Defaults to 6.

        Raises:
            ValueError: If `alpha` is not between 0 and 1.
        """
        SimpleCornerEnvRS10.__init__(self, pilot=pilot)

        self.observation_space = self._get_observation_space()

        Pilot.__init__(self, model, seed, alpha, action_space)

        # Initialize the policy network
        self.policy_net = None
        self.set_model(model)

        # Debugging print statements (can be removed in production)
        print("action_space")
        print(self.action_space)

    def set_model(self, model: PPO):
        """
        Given a PPO model, extracts its policy network.

        Args:
            model (PPO): The PPO model providing policy actions.
        """
        if model is not None:
            self.policy_net = model.policy

    def _get_observation_space(self):
        """
        Defines the observation space for the environment, including an additional dimension for
        the pilot's chosen action.

        Returns:
            Box: The observation space represented as a continuous range.
        """
        return Box(low=-1, high=1, shape=(11,), dtype=np.float64)

    def __get_index_from_value(self, value, arreglo_tensor):
        """
        Finds the index of a given value in a tensor.

        Args:
            value (int): The value to find.
            arreglo_tensor (Tensor): A tensor containing action preferences.

        Returns:
            int: The index of the value in the tensor, or -1 if not found.
        """
        for i in range(len(arreglo_tensor)):
            if arreglo_tensor[i].item() == value:
                return i
        return -1

    def choose_action(self, obs):
        """
        Selects an action based on the observation, blending decisions from the pilot and co-pilot.

        Args:
            obs (array-like): The observation input to the environment.

        Returns:
            tuple:
                - action_to_apply (int): The final action chosen.
                - _states (list): Additional state information (if applicable).
        """
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)

        # Compute logits and probabilities from the policy network
        with torch.no_grad():
            latent_pi = self.policy_net.features_extractor(obs_tensor)
            logits = self.policy_net.mlp_extractor.policy_net(latent_pi)
            action_logits = self.policy_net.action_net(logits)

        action_logits = action_logits.squeeze(0)
        probabilities = F.softmax(action_logits, dim=-1)

        # Debugging outputs
        print("probabilities")
        print(probabilities)

        # Sort actions by preference
        action_preferences = torch.argsort(action_logits, descending=True)
        print("action_preferences")
        print(action_preferences)

        index_pilot = self.__get_index_from_value(self.pilot_action, action_preferences)
        copilot_action = action_preferences[0].item()
        _states = []
        action_to_apply = 0

        # Blend pilot and co-pilot actions based on probabilities and alpha
        if probabilities[index_pilot] >= ((1 - self._alpha) * probabilities[copilot_action]):
            action_to_apply = self.pilot_action
            print("PILOT")
        else:
            action_to_apply = copilot_action
            print("COPILOT")

        return action_to_apply, _states
