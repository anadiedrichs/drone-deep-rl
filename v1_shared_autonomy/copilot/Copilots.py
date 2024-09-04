"""
More runners for discrete RL algorithms can be added here.
"""

import sys
sys.path.append('../utils')
from utilities import *
from pid_controller import *
from gym.spaces import Box
from CrazyflieDrone import CornerEnv
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
        self.__alpha = 0.5
        self.__seed = seed
        np.random.seed(self.__seed)

    def set_alpha(self, value):
        if value is not None:
            if value >=0 and value <=1:
                self.alpha = value
            else:
                raise ValueError("alpha must be a coefficient between 0 and 1")

    def set_model(self, model: PPO):

        if model is not None:
            # get the ( neural net ) policy
            self.policy_net = model.policy

    def _get_observation_space(self):
        """
        We add one observation to the DroneRobotSupervisor observation list:
        the base-pilot's chosen action.
        """

        # [roll, pitch, yaw_rate, v_x, v_y, self.altitude ,self.range_front_value, self.range_back_value ,self.range_right_value, self.range_left_value ] #4

        #self.observation_space = Box(low=np.array([- pi, -pi / 2, - pi, 0, 0, 0.1, 2.0, 2.0, 2.0, 2.0]),
        #                             high=np.array([pi, pi / 2, pi, 10, 10, 5, 2000, 2000, 2000, 2000]),
        #
        return Box(low=-1, high=1, shape=(11,), dtype=np.float64)

    def __get_index_from_value(self,value, arreglo_tensor):
        """
        Regresa el índice del elemento de action_preferences al que es igual el valor dado.

        Args:
          value: El valor entero a buscar.
          arreglo_tensor: El tensor con los índices de las acciones ordenadas por preferencia.

        Returns:
          El índice del elemento de action_preferences al que es igual el valor dado.
        """
        for i in range(len(arreglo_tensor)):
            if arreglo_tensor[i].item() == value:
                return i
        return -1  # Si el valor no se encuentra en action_preferences

    def choose_action(self, obs):

        # print("OBSERVATIONS")
        # print(obs)
        # print("shape")
        # print(obs.shape)

        # Convertir la observación a un tensor y agregar una dimensión adicional
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)

        #print(obs_tensor)
        #print(obs_tensor.shape)
        #print(obs_tensor.dim())

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
        print("probabilities")
        print(probabilities)
        # Ordenar las acciones por preferencia
        action_preferences = torch.argsort(action_logits, descending=True)
        print("action_preferences")
        print(action_preferences)

        index_pilot = self.__get_index_from_value(self.pilot_action, action_preferences)
        copilot_action = action_preferences[0].item()
        _states = []
        action_to_apply = 0

        if probabilities[index_pilot] >= ((1-self.alpha) * probabilities[copilot_action]):
            action_to_apply = self.pilot_action
            print("PILOTO")
        else:
            action_to_apply = copilot_action
            print("COPILOTO")

        return action_to_apply, _states
