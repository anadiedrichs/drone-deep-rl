from controller import Supervisor

import pandas as pd

import pandas as pd

# Función para leer la rotación desde un archivo CSV utilizando pandas e ignorar comentarios
def leer_rotacion_desde_archivo(ruta_archivo):
    try:
        # Leer el archivo CSV, ignorando comentarios que comiencen con '#'
        df = pd.read_csv(ruta_archivo, comment='#')

        # Verificar si las columnas correctas están presentes
        columnas_esperadas = ['x', 'y', 'z', 'rotation']
        if not all(col in df.columns for col in columnas_esperadas):
            raise ValueError(f"El archivo no contiene las columnas esperadas: {columnas_esperadas}")

        # Extraer los valores de la primera fila
        rotacion = df.iloc[0][['x', 'y', 'z', 'rotation']].tolist()

        # Verificar que los valores son correctos (4 valores)
        if len(rotacion) != 4:
            raise ValueError(f"Se esperaban 4 valores en la rotación, pero se encontraron {len(rotacion)}.")

        return rotacion

    except FileNotFoundError:
        print(f"Error: El archivo {ruta_archivo} no se encuentra.")
        return None
    except Exception as e:
        print(f"Error al leer el archivo {ruta_archivo}: {str(e)}")
        return None


# Función para guardar la rotación en un archivo CSV utilizando pandas
def guardar_rotacion_en_archivo(ruta_archivo, rotacion):
    try:
        # Crear un DataFrame con los valores de rotación
        df = pd.DataFrame([rotacion], columns=['x', 'y', 'z', 'rotation'])

        # Guardar el DataFrame en un archivo CSV, sin índice y con el formato deseado
        df.to_csv(ruta_archivo, index=False)

    except Exception as e:
        print(f"Error al guardar el archivo {ruta_archivo}: {str(e)}")


# Función para incrementar la rotación en el eje Z en 45 grados
def incrementar_rotacion_en_z(ruta_archivo):
    rotacion = leer_rotacion_desde_archivo(ruta_archivo)
    # Incrementar el ángulo de rotación en 45 grados (0.785 radianes)
    rotacion[3] += 0.785  # El cuarto valor es el ángulo en radianes
    # Asegurarse de que el ángulo está en el rango [0, 2π]
    rotacion[3] = rotacion[3] % (2 * 3.14159)
    # Escribir la nueva rotación en el archivo
    guardar_rotacion_en_archivo(ruta_archivo, rotacion)


FILE_NAME = "rotation.txt"
# Crear la instancia del Supervisor
supervisor = Supervisor()
time_step = int(supervisor.getBasicTimeStep())

# Obtener el nodo del drone
drone_node = supervisor.getFromDef("MY_CRAZYFLIE")

# Leer la rotación desde el archivo (que ahora está actualizada)
rotacion = leer_rotacion_desde_archivo(FILE_NAME)
if rotacion:
    print(f"Rotación leída: {rotacion}")

# Asignar la rotación al drone
drone_node.getField("rotation").setSFRotation(rotacion)

# Incrementar la rotación en el eje Z y actualizar el archivo
incrementar_rotacion_en_z(FILE_NAME)

# Iniciar la simulación
while supervisor.step(time_step) != -1:
    pass
