from DroneTrajectoryLogger import *

if __name__ == '__main__':
    logger = DroneTrajectoryLogger(model_seed=5, model_total_timesteps=20000,
                                   log_path="./20241208-testSimpleCornerEnvRS10/", n_eval_episodes=20)
    logger.run_experiment()