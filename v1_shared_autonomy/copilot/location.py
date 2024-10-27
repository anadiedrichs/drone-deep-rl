import pandas as pd
import random
import time
# TODO traducir al inglés

# Inicializar el generador de números aleatorios con la hora de la PC
random.seed(time.time())


# Función para generar valores aleatorios entre -1 y 1 para x e y
def generar_ubicacion_aleatoria():
    x = random.uniform(-0.8, 0.8)
    y = random.uniform(-0.8, 0.8)
    return [x, y]


# Función para guardar la ubicación en un archivo CSV utilizando pandas
def guardar_ubicacion_en_archivo(ruta_archivo, ubicacion):
    try:
        # Crear un DataFrame con los valores de ubicación
        df = pd.DataFrame([ubicacion], columns=['x', 'y'])

        # Guardar el DataFrame en un archivo CSV, sin índice y con el formato deseado
        df.to_csv(ruta_archivo, index=False)

    except Exception as e:
        print(f"Error al guardar el archivo {ruta_archivo}: {str(e)}")


# Función para leer la ubicación desde un archivo CSV utilizando pandas
def leer_ubicacion_desde_archivo(ruta_archivo):
    try:
        # Leer el archivo CSV, ignorando comentarios que comienzan con '#'
        df = pd.read_csv(ruta_archivo, comment='#')

        # Verificar si las columnas correctas están presentes
        columnas_esperadas = ['x', 'y']
        if not all(col in df.columns for col in columnas_esperadas):
            raise ValueError(f"El archivo no contiene las columnas esperadas: {columnas_esperadas}")

        # Extraer los valores de la primera fila
        ubicacion = df.iloc[0][['x', 'y']].tolist()

        # Verificar que se han leído exactamente 2 valores
        if len(ubicacion) != 2:
            raise ValueError(f"Se esperaban 2 valores en la ubicación, pero se encontraron {len(ubicacion)}.")

        return ubicacion

    except FileNotFoundError:
        print(f"Error: El archivo {ruta_archivo} no se encuentra.")
        return None
    except Exception as e:
        print(f"Error al leer el archivo {ruta_archivo}: {str(e)}")
        return None
