# Drone Deep Reinforcement Learning

Status: personal repository with experimental & unstable code.

## Introducción

Repositorio creado para alojar los recursos utilizados para el desarrollo de la tesis de Ana Laura Diedrichs, titulada  "Teleoperación asistida de cuadricópteros mediante aprendizaje por refuerzo profundo" de la [Carrera de Especialización en Inteligencia Artificial](https://lse.posgrados.fi.uba.ar/posgrados/especializaciones/inteligencia-artificial) de la Facultad de Ingeniería de la Facultad de Buenos Aires.

Este repositorio contiene el código y recursos necesarios para un simular un prototipo de un sistema de teleoperación asistida en tiempo real para cuadricópteros, basado en técnicas de aprendizaje por refuerzo profundo. Este sistema se desarrolló para la empresa Ekumen con el objetivo de mejorar la seguridad y la eficiencia de sus operaciones en entornos peligrosos o inaccesibles para los humanos.


## Tabla de Contenidos

- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Instalación

Para ejecutar este proyecto localmente, sigue los siguientes pasos:

1. Instala el simulador Webots [siguiendo las instrucciones según tu sistema operativo](https://cyberbotics.com/doc/guide/installation-procedure)
2. Instala Python, versión 3.7 o superior, [según las instrucciones para tu sistema operativo](https://www.python.org/downloads/)
3. (Recomendado, no obligatorio) Instala [PyCharm](https://www.jetbrains.com/es-es/pycharm/download) como IDE de desarrollo.
Webots ofrece un tutorial para integrar el uso el simulador con Pycharm, [enlace](https://cyberbotics.com/doc/guide/using-your-ide#pycharm)
4. Clona el repositorio:
    ```bash
    git clone https://github.com/anadiedrichs/drone-deep-rl.git
    cd drone-deep-rl
    ```
5. Crea y activa un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

6. Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Cada programa o ejemplo tiene su propio README explicando su uso.

## Estructura del Proyecto

- `0-tutorial/`: Contiene ejemplos y explicaciones del uso de Webots, Gym.Env, entre otros.
- `v1-shared-autonomy/`: Desarrollo del enfoque de autonomía compartida.
- `requirements.txt`: Archivo de dependencias necesarias para el proyecto.
- `README.md`: Este archivo.

## Licencia

Este proyecto está bajo la Licencia Apache-2.0. 

Para más detalles, revisa el archivo [LICENSE](LICENSE).

## Memoria

[Documento Memoria técnica versión al 2024-11-01](https://www.dropbox.com/scl/fi/pn8v7tif7i3exfrrzuqmh/CEIA_FIUBA_2024_memoria_20241101.pdf?rlkey=0vzd0gm51qyx5kt28umotmb3h&dl=0)

## Contacto

Para preguntas, sugerencias o comentarios, puedes contactar a:

- Ana Diedrichs - ana (dot) diedrichs (at) docentes.frm.utn.edu.ar

