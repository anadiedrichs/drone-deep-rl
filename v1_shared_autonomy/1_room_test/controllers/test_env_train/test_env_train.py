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

    def is_in_the_corner(self,min_distance=500):
        """
            Return True if the distance sensors detect a corner at min_distance value
        """

        assert min_distance is not None
        assert min_distance < 2000
        assert min_distance > 10

        return bool((self.dist_front <= min_distance and self.dist_left <= min_distance) or \
             (self.dist_front <= min_distance and self.dist_right <= min_distance) or \
             (self.dist_back <= min_distance and self.dist_left <= min_distance) or \
             (self.dist_back <= min_distance and self.dist_right <= min_distance))


    def get_reward(self, action=6):

        """
                
        """
        reward = 0
        action = int(action)
        r_avoid_obstacle = 0

        # Calculate the minimum distance to the walls
        dist_min = min(self.dist_front, self.dist_back, self.dist_right, self.dist_left)
        # dist_average = (self.dist_front + self.dist_back + self.dist_right + self.dist_left) / 4
        print("DEBUG  dist_min " + str(dist_min))

        # dist_min in (400,500)
        if(dist_min < 500 and dist_min > 400 and self.is_in_the_corner(500)):
            reward=10
        elif(dist_min > 300 and self.is_in_the_corner(400)):
            reward=20
        elif (dist_min > 200 and self.is_in_the_corner(300)):
            reward = 30
        elif(self.is_in_the_corner(200)):
            reward = 40

        # penalize if the drone is not near a corner
        if reward == 0:
            reward = -0.1 * dist_min

        # Reward for every step the episode hasn't ended
        print("Reward value " + str(reward))

        # update cummulative reward
        self.episode_score += reward
        print("Episode score " + str(self.episode_score))

        # Calcular valores mínimo y máximo de recompensa
        # Dependiente del escenario
        min_reward = -100  #
        max_reward = 40  #

        # Normalizar la recompensa
        normalized_reward = normalize_to_range(reward, min_reward, max_reward, -1, 1)
        # print("DEBUG normalized reward " + str(normalized_reward))
        return normalized_reward

    def is_done(self):
        """
        Return True when:
         * the drone reach a corner
        """
        done = False
        # if the drone reach a corner
        if bool((self.dist_front <= 100 and self.dist_left <= 100) or \
                (self.dist_front <= 100 and self.dist_right <= 100) or \
                (self.dist_back <= 100 and self.dist_left <= 100) or \
                (self.dist_back <= 100 and self.dist_right <= 100)):
            done = True
            self.is_success = True
        else:
            # analyze reward threshold
            if self.episode_score <= -1_000_000 or self.episode_score > 100_000:
                done = True
                self.is_success = False

        return done


def test_env():
    # Initialize the environment
    env = PilotRoom1()
    #env = DummyVecEnv([lambda: env])
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
    print("INICIO ENTRENAMIENTO")
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
