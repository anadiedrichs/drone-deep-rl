import numpy as np
import torch
from stable_baselines3 import PPO
import torch.nn.functional as F

# model file path
model_path = "../test_env_train/logs-2024-08-05_1/ppo_model_pilot_room_1"  # os.path.join(log_path,"ppo_model_pilot_room_1")
# Load a model
model = PPO.load(model_path, print_system_info=True)
# Extraer la red neuronal
policy_net = model.policy

# Observación de entrada:
# Debe ser preprocesada en el mismo formato que durante el entrenamiento

obs = np.array([0.1 for _ in range(10)])
print(obs.shape)
# Convertir la observación a un tensor y agregar una dimensión adicional
obs_tensor = torch.tensor(obs).float().unsqueeze(0)

print(obs_tensor.shape)
print(obs_tensor.dim())
# Pasar la observación a la red de política
with torch.no_grad():
    latent_pi = policy_net.features_extractor(obs_tensor)

    # Aplicar la capa de acción
    logits = policy_net.mlp_extractor.policy_net(latent_pi)
    action_logits = policy_net.action_net(logits)

# Eliminar la dimensión adicional de batch si quieres ver los resultados como un vector simple
action_logits = action_logits.squeeze(0)

# convertir logits en probabilidades
probabilities = F.softmax(action_logits, dim=-1)

# Ordenar las acciones por preferencia
action_preferences = torch.argsort(action_logits, descending=True)

prob_sorted, _ = torch.sort(probabilities, descending=True)


# print("logits")
# print(logits)
# print("action_logits")
# print(action_logits)
print("probabilities")
print(probabilities)
print("action_preferences")
print(action_preferences)
# print("sum prob ")
# print(torch.sum(probabilities))
print("sorted prob ")
print(prob_sorted)
print("cum sum sorted prob ")
cumsumvar = torch.cumsum(prob_sorted,0)
print(torch.cumsum(prob_sorted,0))
print("filter numpy array ")
print(cumsumvar < 0.5)
print("selected actions")
print(action_preferences[cumsumvar < 0.5])
print(np.isin(1, action_preferences[cumsumvar < 0.5]))
