"""

Create a custom environment.

Then we test the different methods

"""

import sys
from controller import Supervisor

sys.path.append('../../../utils')
from utilities import *
from pid_controller import *

sys.path.append('../../../copilot')
from CrazyflieDrone import *

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import *

class PilotRoom1(DroneOpenAIGymEnvironment):

    def __init__(self):
        super().__init__()

    def get_reward(self, action=6):

        """
                
        """
        reward = 0
        action = int(action)
        r_avoid_obstacle = 0

        # 2000 mm es el máximo, 100 mm = 10 cm
        # if the drone reach a corner
        if bool((self.dist_front <= 100 and self.dist_left <= 100) or \
                (self.dist_front <= 100 and self.dist_right <= 100) or \
                (self.dist_back <= 100 and self.dist_left <= 100) or \
                (self.dist_back <= 100 and self.dist_right <= 100)):

            # drone wins a big reward
            reward = 10
        else:

            # be near an obstacle
            too_close = (self.dist_front <= 100 or self.dist_left <= 100 or \
                         self.dist_right <= 100 or self.dist_back <= 100)

            if too_close:
                # penalize to be too close an obstacle or wall
                r_avoid_obstacle = -1
            else:
                # reward to keep going
                r_avoid_obstacle = 1

        reward = reward + r_avoid_obstacle

        # Calcular valores mínimo y máximo de recompensa
        # Dependiente del escenario
        min_reward = -1  #
        max_reward = 10  #

        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)

        # Reward for every step the episode hasn't ended
        # print("DEBUG Reward value " + str(reward))
        # print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

    def is_done(self):
        """
        Return True when:
         * the drone reach a corner
        """

        # if the drone reach a corner
        if bool((self.dist_front <= 100 and self.dist_left <= 100) or \
                (self.dist_front <= 100 and self.dist_right <= 100) or \
                (self.dist_back <= 100 and self.dist_left <= 100) or \
                (self.dist_back <= 100 and self.dist_right <= 100)):
            return True

        return False


def test_env():
    # Initialize the environment
    env = PilotRoom1()

    tmp_path = "./logs/"

    env = Monitor(env, filename=tmp_path, info_keywords=("is_success",))


    # model to train: PPO
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1, seed=5, tensorboard_log=tmp_path)
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard", "log"])
    model.set_logger(new_logger)
    # access to tensorboard logs typing in your terminal:
    # tensorboard --logdir ./logs/
    # CALLBACKS

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=100,
    #                             callback_after_eval=stop_train_callback,
    #                             best_model_save_path="./logs/best-model/",
    #                             deterministic=True, render=False,
    #                             verbose=1)

    # Stops training when the model reaches the maximum number of episodes
    # callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=50, verbose=1)
    # Create the callback list
    # callback_list = CallbackList([eval_callback, callback_max_episodes])

    # Almost infinite number of timesteps, but the training will stop early
    # as soon as the number of consecutive evaluations without model
    # improvement is greater than 3

    # start training
    # model.learn(total_timesteps=100_000)
    model.learn(total_timesteps=50_000, # callback=eval_callback,
                progress_bar=True)

    # Save the learned model
    model.save("./logs/ppo_model_pilot_room_1")


    # Test the trained agent
    # using the env
    obs = env.reset()
    n_steps = 200
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", done)

        if done:
            # Note that the env resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break

    print("EXPERIMENT ended")
    # close Webots simulator
    # env.simulationQuit(0)



if __name__ == '__main__':
    test_env()
