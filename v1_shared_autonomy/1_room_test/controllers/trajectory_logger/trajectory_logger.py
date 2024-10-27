from DroneTrajectoryLogger import *

if __name__ == '__main__':
    logger = DroneTrajectoryLogger(model_seed=10, model_total_timesteps=20000, log_path="./custom_logs/", n_eval_episodes=10)
    logger.run_experiment()