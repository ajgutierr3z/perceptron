# perceptron_and.py
"""
Práctica de Perceptrón: Reconociendo la compuerta lógica AND
Autor: Ajgutierr3z
Descripción: Entrena un perceptrón simple para simular el comportamiento de una compuerta AND
"""

import numpy as np

# -------------------------
# 1. Datos de entrada y salida esperada
# -------------------------

# Entradas (X1, X2)
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas esperadas (Y)
salidas = np.array([
    [0],
    [0],
    [0],
    [1]
])

# -------------------------
# 2. Inicialización de pesos y bias
# -------------------------

# Inicializa pesos y bias aleatoriamente
pesos = np.random.rand(2, 1)
bias = np.random.rand(1)

# Tasa de aprendizaje
tasa_aprendizaje = 0.1

# -------------------------
# 3. Función de activación
# -------------------------

def activacion(x):
    """Función escalón binaria."""
    return 1 if x >= 1 else 0

# -------------------------
# 4. Entrenamiento del perceptrón
# -------------------------

print("Entrenando el perceptrón para compuerta AND...\n")

for epoca in range(10):  # Número de épocas
    print(f"Época {epoca + 1}")
    for x, y_esperado in zip(entradas, salidas):
        salida_neta = np.dot(x, pesos) + bias
        salida_obtenida = activacion(salida_neta)

        # Error y ajuste
        error = y_esperado - salida_obtenida

        # Actualización de pesos y bias
        pesos += tasa_aprendizaje * error * x.reshape(2, 1)
        bias += tasa_aprendizaje * error

        print(f" Entrada: {x}, Esperado: {y_esperado[0]}, Obtenido: {salida_obtenida}")
    print("-" * 40)

# -------------------------
# 5. Pruebas del perceptrón ya entrenado
# -------------------------

print("\nResultado final después del entrenamiento:")
for x in entradas:
    salida_neta = np.dot(x, pesos) + bias
    salida_obtenida = activacion(salida_neta)
    print(f" Entrada: {x} → Salida: {salida_obtenida}")

# -------------------------
# Fin del programa
# -------------------------
