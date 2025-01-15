from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
import numpy as np

class Pilot:
    """
    Represents a generic pilot interface that integrates with reinforcement learning models (PPO or BC) 
    to determine actions based on observations. This class is intended to be extended for specific behaviors 
    such as NoisyPilot or LaggyPilot.
    """

    def __init__(self, model, seed=1, alpha=0.5, action_space=6):
        """
        Initializes the Pilot object.

        Args:
            model (PPO or BC): A reinforcement learning model, either a PPO or BC instance.
            seed (int, optional): Seed for the random number generator. Defaults to 1.
            alpha (float, optional): A parameter for child classes to define specific behavior. Defaults to 0.5.
            action_space (int, optional): Size of the action space. Defaults to 6.

        Raises:
            ValueError: If the provided model is not an instance of PPO or BC.
        """
        self._check_model(model)
        self._validate_alpha(alpha)
        self._rng = np.random.default_rng(seed)  # Independent random generator for this instance
        self._action_space = action_space


    def _check_model(self, model):
        """
        Validates the provided model to ensure it is a non-null instance of PPO or BC.

        Args:
            model: The model to validate.

        Raises:
            ValueError: If the model is None or not an instance of PPO or BC.
        """
        if model is None or not isinstance(model, (PPO, BC)):
            raise ValueError("The model must be a non-null instance of PPO or BC.")
        else:
            self._model = model

    def _validate_alpha(self, alpha):
            """
            Validates that alpha is a float between 0 and 1.

            Args:
                alpha (float): The alpha value to validate.

            Raises:
                ValueError: If alpha is not between 0 and 1.
            """
            if not ((0 <= alpha) and (alpha <= 1)):
                raise ValueError("Alpha must be a float between 0 and 1.")

    def get_action_from_model(self, obs):
        """
        Predicts the action for a given observation using the associated model.

        Args:
            obs (array-like): The observation input to the model.

        Returns:
            tuple: A tuple containing:
                - action: The predicted action from the model.
                - _states: Additional state information (if applicable).

        Raises:
            ValueError: If the model is not an instance of PPO or BC.
        """
        action = None
        _states = []

        if isinstance(self._model, PPO):
            action, _states = self._model.predict(obs)
        elif isinstance(self._model, BC):
            action, _states = self._model.policy.predict(obs, deterministic=True)
        else:
            raise ValueError("Model must be of type PPO or BC.")

        return action, _states

    def choose_action(self, obs):
        #TODO check type for obs
        """
        Abstract method to be implemented in child classes for selecting actions based on observations.

        Args:
            obs (array-like): The observation input to the model. 

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError("Please implement this method in a child class.")
