import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de la carpeta que contiene los archivos CSV
folder_path = '../1_room_test/controllers/trajectory_logger/20241208-testSimpleCornerEnvRS10/'  #

print(os.path.join(folder_path, 'trajectory_*.csv'))
# Buscar todos los archivos que comienzan con 'trajectory_'
csv_files = glob.glob(os.path.join(folder_path, 'trajectory_*'))

# Verificar si se encontraron archivos
if not csv_files:
    print("No se encontraron archivos con el prefijo 'trajectory_'.")
else:
    # Crear la figura
    plt.figure(figsize=(10, 8))

    for idx, file in enumerate(csv_files):
        # Leer el archivo CSV
        data = pd.read_csv(file)

        # Verificar que las columnas requeridas existan
        if not all(col in data.columns for col in ['time', 'x', 'y', 'z']):
            print(f"El archivo {file} no contiene las columnas necesarias. Se omitirá.")
            continue

        # Graficar la trayectoria
        plt.plot(data['x'], data['y'], label=f'Trayectoria {idx + 1}')

    # Configurar etiquetas y leyenda
    plt.title('Trayectorias')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.legend()
    plt.grid()
    #plt.show()

    output_file = "trayectorias.png"  # Nombre del archivo de salida
    plt.savefig(output_file)
    print(f"Gráfica guardada en {output_file}")

