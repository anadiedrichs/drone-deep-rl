"""
More runners for discrete RL algorithms can be added here.
"""

import sys

from gym.spaces import Box, Discrete

from v1_shared_autonomy.copilot.CrazyflieDrone import CornerEnv

sys.path.append('../utils')
from utilities import *
from pid_controller import *

sys.path.append('../pilots')

from pilot import *
from v1_shared_autonomy.pilots.pilot import Pilot

import gym  #nasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn.functional as F


class CopilotCornerEnv(CornerEnv, Pilot):

    def __init__(self, seed, model: PPO):
        super().__init__()

        if model is None or not isinstance(model, PPO):
            raise ValueError("The model must be a non-null instance of PPO.")

        self.seed = seed
        np.random.seed(self.seed)
        self.pilot_action = None
        self.model = model
        # Extraer la red neuronal
        self.policy_net = self.model.policy

    def get_observation_space(self):
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

        # Convertir la observación a un tensor y agregar una dimensión adicional
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)

        # Pasar la observación a la red de política
        with torch.no_grad():
            latent_pi = self.policy_net.features_extractor(obs_tensor)
            # Aplicar la capa de acción
            logits = self.policy_net.mlp_extractor.policy_net(latent_pi)
            action_logits = self.policy_net.action_net(logits)

        # Eliminar la dimensión adicional de batch
        # si quieres ver los resultados como un vector simple
        action_logits = action_logits.squeeze(0)

        # convertir logits en probabilidades
        probabilities = F.softmax(action_logits, dim=-1)

        # Ordenar las acciones por preferencia
        action_preferences = torch.argsort(action_logits, descending=True)

        prob_sorted, _ = torch.sort(probabilities, descending=True)

        cumsumvar = torch.cumsum(prob_sorted, 0)

        selected_actions = action_preferences[cumsumvar < alpha_prob]

        if  np.isin(self.pilot_action, selected_actions):
            return self.pilot_action
        else:
            return selected_actions[0]
