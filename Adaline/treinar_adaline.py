import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TREINAMENTO = np.array(
    [
        [0.4329, -1.3719, 0.7022, -0.8535, 1.0000],
        [0.3024, 0.2286, 0.8630, 2.7909, -1.0000],
        [0.1349, -0.6445, 1.0530, 0.5687, -1.0000],
        [0.3374, -1.7163, 0.3670, -0.6283, -1.0000],
        [1.1434, -0.0485, 0.6637, 1.2606, 1.0000],
        [1.3749, -0.5071, 0.4464, 1.3009, 1.0000],
        [0.7221, -0.7587, 0.7681, -0.5592, 1.0000],
        [0.4403, -0.8072, 0.5154, -0.3129, 1.0000],
        [-0.5231, 0.3548, 0.2538, 1.5776, -1.0000],
        [0.3255, -2.0000, 0.7112, -1.1209, 1.0000],
        [0.5824, 1.3915, -0.2291, 4.1735, -1.0000],
        [0.1340, 0.6081, 0.4450, 3.2230, -1.0000],
        [0.1480, -0.2988, 0.4778, 0.8649, 1.0000],
        [0.7359, 0.1869, -0.0872, 2.3584, 1.0000],
        [0.7115, -1.1469, 0.3394, 0.9573, -1.0000],
        [0.8251, -1.2840, 0.8452, 1.2382, -1.0000],
        [0.1569, 0.3712, 0.8825, 1.7633, 1.0000],
        [0.0033, 0.6835, 0.5389, 2.8249, -1.0000],
        [0.4243, 0.8313, 0.2634, 3.5855, -1.0000],
        [1.0490, 0.1326, 0.9138, 1.9792, 1.0000],
        [1.4276, 0.5331, -0.0145, 3.7286, 1.0000],
        [0.5971, 1.4865, 0.2904, 4.6069, -1.0000],
        [0.8475, 2.1479, 0.3179, 5.8235, -1.0000],
        [1.3967, -0.4171, 0.6443, 1.3927, 1.0000],
        [0.0044, 1.5378, 0.6099, 4.7755, -1.0000],
        [0.2201, -0.5668, 0.0515, 0.7829, 1.0000],
        [0.6300, -1.2480, 0.8591, 0.8093, -1.0000],
        [-0.2479, 0.8960, 0.0547, 1.7381, 1.0000],
        [-0.3088, -0.0929, 0.8659, 1.5483, -1.0000],
        [-0.5180, 1.4974, 0.5453, 2.3993, 1.0000],
        [0.6833, 0.8266, 0.0829, 2.8864, 1.0000],
        [0.4353, -1.4066, 0.4207, -0.4879, 1.0000],
        [-0.1069, -3.2329, 0.1856, -2.4572, -1.0000],
        [0.4662, 0.6261, 0.7304, 3.4370, -1.0000],
        [0.8298, -1.4089, 0.3119, 1.3235, -1.0000],
    ]
)

TESTE = np.array(
    [
        [0.9694, 0.6909, 0.4334, 3.4965],
        [0.5427, 1.3832, 0.6390, 4.0352],
        [0.6081, -0.9196, 0.5925, 0.1016],
        [-0.1618, 0.4694, 0.2030, 3.0117],
        [0.1870, -0.2578, 0.6124, 1.7749],
        [0.4891, -0.5276, 0.4378, 0.6439],
        [0.3777, 2.0149, 0.7423, 3.3932],
        [1.1498, -0.4067, 0.2469, 1.5866],
        [0.9325, 1.0950, 1.0359, 3.3591],
        [0.5060, 1.3317, 0.9222, 3.7174],
        [0.0497, -2.0656, 0.6124, -0.6585],
        [0.4004, 3.5369, 0.9766, 5.3532],
        [-0.1874, 1.3343, 0.5374, 3.2189],
        [0.5060, 1.3317, 0.9222, 3.7174],
        [1.6375, -0.7911, 0.7537, 0.5515],
    ]
)


def sinal(valor):
    return 1 if valor >= 0 else -1


def preparar_entradas(dados):
    x = dados[:, :4]
    bias = np.full((x.shape[0], 1), -1.0)
    return np.hstack((bias, x))


def erro_quadratico_medio(X, d, w):
    erro = d - X @ w
    return np.mean(erro**2) / 2.0


def treinar_adaline(X, d, taxa_aprendizado, precisao, seed, max_epocas=100000):
    rng = np.random.default_rng(seed)
    w_inicial = rng.uniform(0.0, 1.0, X.shape[1])
    w = w_inicial.copy()
    historico_eqm = []
    eqm_anterior = erro_quadratico_medio(X, d, w)

    for epoca in range(1, max_epocas + 1):
        for x_i, d_i in zip(X, d):
            y_i = x_i @ w
            erro_i = d_i - y_i
            w += taxa_aprendizado * erro_i * x_i

        eqm_atual = erro_quadratico_medio(X, d, w)
        historico_eqm.append(eqm_atual)

        if abs(eqm_anterior - eqm_atual) <= precisao:
            return w_inicial, w, epoca, historico_eqm

        eqm_anterior = eqm_atual

    return w_inicial, w, max_epocas, historico_eqm


def classificar(X, pesos):
    campos = X @ pesos
    classes = np.array([sinal(v) for v in campos])
    return campos, classes


def salvar_grafico_eqm(resultados, output_dir):
    plt.figure(figsize=(10, 5.5))
    for resultado in resultados[:2]:
        epocas = range(1, len(resultado["historico_eqm"]) + 1)
        plt.plot(epocas, resultado["historico_eqm"], linewidth=2, label=resultado["nome"])

    plt.title("EQM por Epoca nos Treinamentos T1 e T2")
    plt.xlabel("Epoca")
    plt.ylabel("Erro quadratico medio")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    caminho = os.path.join(output_dir, "eqm_t1_t2.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    return caminho


def salvar_grafico_classificacao(campos_por_treinamento, output_dir):
    media_campos = np.mean(np.column_stack(campos_por_treinamento), axis=1)
    amostras = np.arange(1, len(media_campos) + 1)
    cores = ["#16a34a" if valor >= 0 else "#dc2626" for valor in media_campos]

    plt.figure(figsize=(11, 5.5))
    plt.bar(amostras, media_campos, color=cores)
    plt.axhline(0, color="#111827", linewidth=1.3)
    plt.title("Campo Local Medio nas Amostras de Teste")
    plt.xlabel("Amostra")
    plt.ylabel("Media de v = X.W nos 5 treinamentos")
    plt.xticks(amostras)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(output_dir, "classificacao_teste.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    return caminho


def linha_tabela_pesos(resultado):
    valores = [
        resultado["nome"],
        *[f"{v:.6f}" for v in resultado["w_inicial"]],
        *[f"{v:.6f}" for v in resultado["w_final"]],
        str(resultado["epocas"]),
    ]
    return "| " + " | ".join(valores) + " |"


def main():
    taxa_aprendizado = 0.0025
    precisao = 1e-6
    seeds = [101, 202, 303, 404, 505]
    output_dir = os.path.dirname(os.path.abspath(__file__))

    X_treino = preparar_entradas(TREINAMENTO)
    d_treino = TREINAMENTO[:, 4]
    X_teste = preparar_entradas(TESTE)

    resultados = []
    campos_por_treinamento = []
    classes_por_treinamento = []

    for indice, seed in enumerate(seeds, start=1):
        w_inicial, w_final, epocas, historico_eqm = treinar_adaline(
            X_treino,
            d_treino,
            taxa_aprendizado=taxa_aprendizado,
            precisao=precisao,
            seed=seed,
        )
        campos, classes = classificar(X_teste, w_final)
        campos_por_treinamento.append(campos)
        classes_por_treinamento.append(classes)
        resultados.append(
            {
                "nome": f"T{indice}",
                "w_inicial": w_inicial,
                "w_final": w_final,
                "epocas": epocas,
                "historico_eqm": historico_eqm,
                "eqm_final": historico_eqm[-1],
            }
        )

    grafico_eqm = salvar_grafico_eqm(resultados, output_dir)
    grafico_classificacao = salvar_grafico_classificacao(campos_por_treinamento, output_dir)

    print("Tabela de treinamentos:")
    print("| Treinamento | w0 inicial | w1 inicial | w2 inicial | w3 inicial | w4 inicial | w0 final | w1 final | w2 final | w3 final | w4 final | Epocas |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for resultado in resultados:
        print(linha_tabela_pesos(resultado))

    print("\nClassificacao das amostras de teste:")
    print("| Amostra | x1 | x2 | x3 | x4 | y(T1) | y(T2) | y(T3) | y(T4) | y(T5) | Comando medio |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    matriz_classes = np.column_stack(classes_por_treinamento)
    for i, linha in enumerate(TESTE, start=1):
        classes = matriz_classes[i - 1]
        comando = "Valvula B" if np.mean(classes) >= 0 else "Valvula A"
        valores = [
            str(i),
            *[f"{v:.4f}" for v in linha],
            *[str(int(v)) for v in classes],
            comando,
        ]
        print("| " + " | ".join(valores) + " |")

    print("\nGraficos gerados:")
    print(grafico_eqm)
    print(grafico_classificacao)


if __name__ == "__main__":
    main()
