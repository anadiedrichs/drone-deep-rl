import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
import csv
def normalize_to_range(value, min_val, max_val, new_min, new_max, clip=False):
    """
    Normalizes value to a specified new range by supplying the current range.

    :param value: value to be normalized
    :type value: float
    :param min_val: value's min value, value ∈ [min_val, max_val]
    :type min_val: float
    :param max_val: value's max value, value ∈ [min_val, max_val]
    :type max_val: float
    :param new_min: normalized range min value
    :type new_min: float
    :param new_max: normalized range max value
    :type new_max: float
    :param clip: whether to clip normalized value to new range or not, defaults to False
    :type clip: bool, optional
    :return: normalized value ∈ [new_min, new_max]
    :rtype: float
    """
    value = float(value)
    min_val = float(min_val)
    max_val = float(max_val)
    new_min = float(new_min)
    new_max = float(new_max)

    if clip:
        return np.clip((new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max, new_min, new_max)
    else:
        return (new_max - new_min) / (max_val - min_val) * (value - max_val) + new_max


def get_distance_from_target(robot_node, target_node):
    robot_coordinates = robot_node.getField('translation').getSFVec3f()
    target_coordinate = target_node.getField('translation').getSFVec3f()

    dx = robot_coordinates[0] - target_coordinate[0]
    dy = robot_coordinates[1] - target_coordinate[1]
    distance_from_target = math.sqrt(dx * dx + dy * dy)
    return distance_from_target


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

# tensorboard hyperparameter logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            # TODO how to save ls_rate
            #lr is a function
            #"learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda":self.model.gae_lambda,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "seed": self.model.seed,
            "target_kl": self.model.target_kl,
            "ent_coef":self.model.ent_coef,
            "total_timesteps":self.model._total_timesteps,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean":0.0,
            "train/value_loss": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

def save_experiment_time(start_time,end_time,file_name):
    duration = end_time - start_time
    # Extraer la duración en minutos y segundos
    duration_in_seconds = duration.total_seconds()
    minutes = int(duration_in_seconds // 60)
    seconds = int(duration_in_seconds % 60)
    # Crear un archivo CSV y guardar la duración
    with open(str(file_name)+'-experiment_duration.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["min.", "s"])
        writer.writerow([minutes, seconds])

    print(f"Duración del experimento: {minutes} minutos y {seconds} segundos")

####################################################33
# no estoy usando las funciones de aqui abajo

def plot_data(data, x_label, y_label, plot_title, save=False, save_name=None):
    """
    Uses matplotlib to plot data.

    :param data: List of data to plot
    :type data: list
    :param x_label: Label on x axis
    :type x_label: str
    :param y_label: Label on y axis
    :type y_label: str
    :param plot_title: Plot title
    :type plot_title: str
    :param save: Whether to save plot automatically or not, defaults to False
    :type save: bool, optional
    :param save_name: Filename of saved plot, defaults to None
    :type save_name: str, optional
    """
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel=x_label, ylabel=y_label,
           title=plot_title)
    ax.grid()
    if save:
        if save_name is not None:
            fig.savefig(save_name)
        else:
            fig.savefig("figure")
    plt.show()


def convert_to_interval_0_600(numero):
    # Normalizar al rango [0, 1]
    numero_normalizado = (numero + 1) / 2

    # Escalar al rango deseado [0, 600]
    numero_convertido = numero_normalizado * 600

    return numero_convertido
