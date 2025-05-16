import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activate(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights)
        return self.activate(weighted_sum)

def plot_decision_boundary(X, y, perceptron):
    x_min, x_max = min(X[:, 0]) - 1, max(X[:, 0]) + 1
    y_min, y_max = min(X[:, 1]) - 1, max(X[:, 1]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Exemplo de uso com dados Iris
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Carregando o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data[:100, :2]  # Pegue apenas as primeiras 100 amostras e as duas primeiras características.
y = (iris.target[:100] != 0) * 1  # Transforme os rótulos em uma classificação binária (setosa ou não-setosa).

# Divida o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crie e treine o Perceptron
perceptron = Perceptron(input_size=2)
perceptron.train(X_train, y_train)

# Faça previsões no conjunto de teste
predictions = [perceptron.predict(x) for x in X_test]

# Avalie a precisão do Perceptron
accuracy = np.mean(predictions == y_test)
print(f"Acurácia do Perceptron: {accuracy * 100}%")

# Visualize o resultado da classificação
plot_decision_boundary(X_train, y_train, perceptron)








# Este código implementa um Perceptron simples, treina-o no conjunto de dados Iris e visualiza o resultado da classificação.
# O exemplo usa apenas duas características do conjunto de dados Iris para fins de visualização, mas você pode usar mais características, dependendo do seu conjunto de dados de entrada.
