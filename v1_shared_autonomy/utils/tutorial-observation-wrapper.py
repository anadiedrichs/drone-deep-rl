import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv  # Cambio en la importación
from gymnasium import ObservationWrapper

# Definir el wrapper personalizado para añadir ruido aditivo a las observaciones
class NoisyObservationWrapper(ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std  # Desviación estándar del ruido

    def observation(self, obs):
        # Sumar ruido gaussiano a las observaciones
        noisy_obs = obs + np.random.normal(0, self.noise_std, size=obs.shape)
        # Limitar las observaciones al rango [-1, 1] (si es necesario)
        noisy_obs = np.clip(noisy_obs, -1, 1)
        return noisy_obs

# Crear el entorno base (ejemplo con CartPole, puedes usar tu propio entorno)
env = gym.make('CartPole-v1')

# Aplicar el wrapper de ruido aditivo
noisy_env = NoisyObservationWrapper(env, noise_std=0.05)

# Crear el vector de entorno para usar con stable-baselines3
vec_env = DummyVecEnv([lambda: noisy_env])  # Ahora se usa el DummyVecEnv de stable-baselines3.common.vec_env

# Entrenar un modelo PPO en este entorno con ruido en las observaciones
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Testear el modelo en el entorno con ruido
obs = vec_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
