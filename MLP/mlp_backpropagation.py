import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class MLPBackpropagation:
    """Rede perceptron multicamadas com uma camada escondida e ligacao direta entrada-saida."""

    def __init__(self, n_inputs, n_hidden, learning_rate=0.5, seed=42):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        # Convencao do enunciado:
        # W1: pesos entre entrada e camada escondida
        # W2: pesos entre camada escondida e saida
        # W3: pesos entre entrada e saida
        limit_w1 = np.sqrt(6.0 / (n_inputs + n_hidden))
        limit_w2 = np.sqrt(6.0 / (n_hidden + 1))
        limit_w3 = np.sqrt(6.0 / (n_inputs + 1))

        self.W1 = self.rng.uniform(-limit_w1, limit_w1, (n_inputs, n_hidden))
        self.W2 = self.rng.uniform(-limit_w2, limit_w2, (n_hidden, 1))
        self.W3 = self.rng.uniform(-limit_w3, limit_w3, (n_inputs, 1))
        self.b1 = np.zeros((1, n_hidden))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1.0 - y)

    def forward(self, X):
        net_hidden = X @ self.W1 + self.b1
        hidden_output = self.sigmoid(net_hidden)

        net_output = hidden_output @ self.W2 + X @ self.W3 + self.b2
        final_output = self.sigmoid(net_output)

        return {
            "net_hidden": net_hidden,
            "hidden_output": hidden_output,
            "net_output": net_output,
            "final_output": final_output,
        }

    def train_epoch(self, X, d):
        cache = self.forward(X)
        y = cache["final_output"]
        z = cache["hidden_output"]

        error = d - y
        output_delta = error * self.sigmoid_derivative(y)
        hidden_delta = (output_delta @ self.W2.T) * self.sigmoid_derivative(z)

        n_samples = X.shape[0]
        dW2 = (z.T @ output_delta) / n_samples
        dW3 = (X.T @ output_delta) / n_samples
        db2 = np.mean(output_delta, axis=0, keepdims=True)
        dW1 = (X.T @ hidden_delta) / n_samples
        db1 = np.mean(hidden_delta, axis=0, keepdims=True)

        self.W2 += self.learning_rate * dW2
        self.W3 += self.learning_rate * dW3
        self.b2 += self.learning_rate * db2
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1

        mse = np.mean(error**2)
        return mse

    def train(self, X, d, epochs=10000, tolerance=1e-4, verbose_every=1000):
        history = []

        for epoch in range(1, epochs + 1):
            mse = self.train_epoch(X, d)
            history.append(mse)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0):
                print(f"Epoca {epoch:5d} | EQM = {mse:.8f}")

            if mse <= tolerance:
                print(f"Treinamento encerrado na epoca {epoch} com EQM = {mse:.8f}")
                break

        return history

    def predict(self, X):
        return self.forward(X)["final_output"]

    def predict_classes(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)


def plot_error_history(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history, color="#2563eb", linewidth=2)
    plt.title("Evolucao do Erro Quadratico Medio")
    plt.xlabel("Epoca")
    plt.ylabel("EQM")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    path = os.path.join(output_dir, "curva_erro_eqm.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_decision_boundary(model, X, d, output_dir):
    x_min, x_max = -0.25, 1.25
    y_min, y_max = -0.25, 1.25
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    contour = plt.contourf(xx, yy, zz, levels=30, cmap="RdYlBu", alpha=0.82)
    plt.contour(xx, yy, zz, levels=[0.5], colors="#111827", linewidths=2)
    plt.colorbar(contour, label="Saida da MLP")

    class_zero = d.ravel() == 0
    class_one = d.ravel() == 1
    plt.scatter(
        X[class_zero, 0],
        X[class_zero, 1],
        c="#dc2626",
        edgecolors="white",
        linewidths=1.5,
        s=130,
        marker="o",
        label="Classe 0",
    )
    plt.scatter(
        X[class_one, 0],
        X[class_one, 1],
        c="#16a34a",
        edgecolors="white",
        linewidths=1.5,
        s=130,
        marker="^",
        label="Classe 1",
    )
    plt.title("Fronteira de Decisao Aprendida para o XOR")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "fronteira_decisao_xor.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_outputs(d, y, output_dir):
    labels = ["[0,0]", "[0,1]", "[1,0]", "[1,1]"]
    positions = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(positions - width / 2, d.ravel(), width, label="Desejado", color="#0f766e")
    plt.bar(positions + width / 2, y.ravel(), width, label="Previsto", color="#f59e0b")
    plt.axhline(0.5, color="#111827", linestyle="--", linewidth=1.2, label="Limiar 0.5")
    plt.title("Saidas Desejadas x Saidas Previstas")
    plt.xlabel("Padrao de entrada")
    plt.ylabel("Valor de saida")
    plt.xticks(positions, labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "saidas_previstas.png")
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    # Exemplo de treinamento: problema XOR.
    # Cada linha representa um padrao de treinamento.
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    d = np.array([[0.0], [1.0], [1.0], [0.0]])

    mlp = MLPBackpropagation(n_inputs=2, n_hidden=4, learning_rate=1.0, seed=7)
    history = mlp.train(X, d, epochs=20000, tolerance=1e-4, verbose_every=2000)

    print("\nSaidas finais:")
    y = mlp.predict(X)
    classes = mlp.predict_classes(X)
    for x_i, d_i, y_i, c_i in zip(X, d, y, classes):
        print(f"x={x_i.astype(int).tolist()} | d={int(d_i[0])} | y={y_i[0]:.6f} | classe={int(c_i[0])}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    generated_paths = [
        plot_error_history(history, output_dir),
        plot_decision_boundary(mlp, X, d, output_dir),
        plot_outputs(d, y, output_dir),
    ]

    print("\nMatrizes finais:")
    print("W1 =")
    print(np.round(mlp.W1, 6))
    print("W2 =")
    print(np.round(mlp.W2, 6))
    print("W3 =")
    print(np.round(mlp.W3, 6))
    print(f"\nEpocas executadas: {len(history)}")
    print(f"EQM final: {history[-1]:.8f}")
    print("\nGraficos gerados:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
