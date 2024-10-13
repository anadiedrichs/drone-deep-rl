from stable_baselines3 import PPO

# Cargar el modelo previamente entrenado
model = PPO.load("./logs-ent-2024-10-11_2/ppo_model_pilot_room_1")

# Mostrar el modelo
print(model.policy)
# Acceder a los parámetros de la política
params = model.policy.state_dict()

# Mostrar los nombres de los parámetros y sus pesos
for name, weight in params.items():
    print(f"Layer: {name}, Weights: {weight}")
