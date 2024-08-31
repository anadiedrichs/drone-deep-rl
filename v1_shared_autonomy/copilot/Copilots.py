"""
More runners for discrete RL algorithms can be added here.
"""

import sys

from gym.spaces import Box

from CrazyflieDrone import CornerEnv

sys.path.append('../utils')
from utilities import *
from pid_controller import *

sys.path.append('../pilots')

from pilot import *

import gym  #nasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn.functional as F


class CopilotCornerEnv(CornerEnv, Pilot):

    def __init__(self, seed, model: PPO):
        super().__init__()

        self.observation_space = self._get_observation_space()

        # model setup
        self.policy_net = None
        self.set_model(model)

        self.__seed = seed
        np.random.seed(self.__seed)

    def set_model(self, model: PPO):

        if model is not None:
            # get the ( neural net ) policy
            self.policy_net = model.policy

    def _get_observation_space(self):
        """
        We add one observation to the DroneRobotSupervisor observation list:
        the pilot's chosen action.
        """

        # [roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4

        #self.observation_space = Box(low=np.array([- pi, -pi / 2, - pi, 0, 0, 0.1, 2.0, 2.0, 2.0, 2.0]),
        #                             high=np.array([pi, pi / 2, pi, 10, 10, 5, 2000, 2000, 2000, 2000]),
        #
        return Box(low=-1, high=1, shape=(11,), dtype=np.float64)

    def choose_action(self, obs, alpha_prob=0.3):

        print("OBSERVATIONS")
        print(obs)
        print("shape")
        print(obs.shape)

        # Convertir la observación a un tensor y agregar una dimensión adicional
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)

        print(obs_tensor)
        print(obs_tensor.shape)
        print(obs_tensor.dim())

        # Pasar la observación a la red de política
        with torch.no_grad():

            latent_pi = self.policy_net.features_extractor(obs_tensor)
            # Aplicar la capa de acción
            logits = self.policy_net.mlp_extractor.policy_net(latent_pi)
            print("logits")
            print(logits)
            action_logits = self.policy_net.action_net(logits)
            print("action_logits")
            print(action_logits)

        # Eliminar la dimensión adicional de batch
        # si quieres ver los resultados como un vector simple
        action_logits = action_logits.squeeze(0)
        print("action_logits")
        print(action_logits)

        # convertir logits en probabilidades
        probabilities = F.softmax(action_logits, dim=-1)
        print("probabilities")
        print(probabilities)

        # Ordenar las acciones por preferencia
        action_preferences = torch.argsort(action_logits, descending=True)
        print("action_preferences")
        print(action_preferences)

        prob_sorted, _ = torch.sort(probabilities, descending=True)
        print("prob_sorted")
        print(prob_sorted)

        cumsumvar = torch.cumsum(prob_sorted, 0)
        print("cumsumvar")
        print(cumsumvar)

        selected_actions = action_preferences[cumsumvar < alpha_prob]
        print("selected_actions")
        print(selected_actions)
        print("self.pilot_action")
        print(self.pilot_action)
        print(np.isin(self.pilot_action, selected_actions.numpy()))
        if np.isin(self.pilot_action, selected_actions.numpy()):
            return self.pilot_action
        else:
            return selected_actions[0]
