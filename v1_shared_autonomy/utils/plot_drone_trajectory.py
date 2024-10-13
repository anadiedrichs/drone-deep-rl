import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
df = pd.read_csv("./logs-dqn/trayectoria.csv")

# Graficar la trayectoria
plt.figure(figsize=(10, 6))
plt.plot(df["x"], df["y"], label="Trayectoria XY")
plt.xlabel("X (metros)")
plt.ylabel("Y (metros)")
plt.title("Trayectoria del Drone")
plt.legend()
plt.grid()
plt.show()
