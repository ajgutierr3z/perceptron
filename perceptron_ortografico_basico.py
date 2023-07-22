import numpy as np

class PerceptronSpellChecker:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def extract_features(self, word):
        # Representa la palabra como un vector de frecuencia de letras (solo minúsculas)
        features = np.zeros(26)
        for char in word:
            if 'a' <= char <= 'z':
                index = ord(char) - ord('a')
                features[index] += 1
        return features

    def fit(self, X, y):
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return 1 if linear_output >= 0 else 0

    def get_word_label(self, prediction):
        return "Incorrecta" if prediction == 1 else "Correcta"        

# Datos de entrenamiento
X_train = np.array([PerceptronSpellChecker().extract_features('hello'),
                    PerceptronSpellChecker().extract_features('world'),
                    PerceptronSpellChecker().extract_features('apple'),
                    PerceptronSpellChecker().extract_features('banana'),
                    PerceptronSpellChecker().extract_features('heeelloo')
                    ])

y_train = np.array([0, 0, 1, 1, 1])  # 0 para palabras correctamente escritas, 1 para palabras con errores

# Crear el modelo del Perceptrón
spell_checker = PerceptronSpellChecker(learning_rate=0.1, epochs=10)

# Entrenar el modelo
spell_checker.fit(X_train, y_train)

# Datos de prueba
X_test = np.array([PerceptronSpellChecker().extract_features('heeello'),   # palabra con error
                   PerceptronSpellChecker().extract_features('world'),     # palabra sin error
                   PerceptronSpellChecker().extract_features('aplle'),     # palabra con error
                   PerceptronSpellChecker().extract_features('bananana')]) # palabra con error

# Realizar predicciones
predictions = [spell_checker.predict(x) for x in X_test]
print(predictions)  # Salida: [1, 0, 1, 1] (Palabra con error, Palabra sin error, Palabra con error, Palabra con error)

# Obtener resultados de las palabras
results = [(word, spell_checker.get_word_label(pred)) for word, pred in zip(['heeello', 'world', 'aplle', 'bananana'], predictions)]

# Mostrar resultados
print("Resultados:")
for word, label in results:
    print(f"Palabra: '{word}' - Clasificación: {label}")
