## choose_action 

Dicha función es el corazón de la autonomía compartida.

### Comportamiento General:

La función `choose_action` dentro de la clase `CopilotCornerEnv` toma como entrada una observación (`obs`) y elige una acción basada en una política combinada entre un piloto (controlador manual o automático básico) y un copiloto (modelo de inteligencia artificial entrenado).

### La explicación del flujo de choose_action:

1. **Entrada y Preparación**:
   - La observación `obs`, que se recibe como entrada, se convierte en un tensor y se le añade una dimensión extra para que pueda ser procesado por la red neuronal de políticas (`policy_net`). Esto se hace mediante `torch.tensor(obs).float().unsqueeze(0)`.

2. **Extracción de Características**:
   - La observación procesada se pasa a través de una red extractora de características (`features_extractor`), que reduce dimensionalidad o resalta aspectos importantes de la observación original.

3. **Red de Política**:
   - La salida de la red extractora de características se pasa a la red de política (`mlp_extractor.policy_net`), que genera una serie de logits. Los logits son los valores que indican la preferencia de la red por cada acción antes de ser convertidos en probabilidades.

4. **Cálculo de Probabilidades**:
   - Los logits se pasan a una capa de acción (`action_net`), y luego se aplica la función softmax (`F.softmax`) para convertirlos en probabilidades. Estas probabilidades representan la probabilidad estimada para cada posible acción.

5. **Ordenar y Seleccionar Acciones**:
   - Se ordenan las acciones de acuerdo con sus logits, en orden descendente (mayor preferencia primero).
   - Se ordenan las probabilidades y se calcula la suma acumulada (`torch.cumsum`). Este cálculo determina el conjunto de acciones que, al sumarlas, dan un valor acumulado menor que `alpha_prob` (en este caso, 0.3).

6**Elección de la acción**:
   - Aquí viene la parte interesante: el programa compara la acción que el piloto básico hubiera elegido (`self.pilot_action`) con la acción más probable sugerida por el copiloto (que es la primera en la lista de `action_preferences`).
   - Si la probabilidad de la acción del piloto es suficientemente alta (según un valor de mezcla `alpha`), entonces se elige la acción del piloto. De lo contrario, se elige la acción sugerida por el copiloto.

7**Devolución**:
   - Finalmente, la función devuelve la acción seleccionada y un estado vacío (que en este caso no se está utilizando).

