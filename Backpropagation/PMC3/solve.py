import re
import numpy as np
import matplotlib.pyplot as plt
import time

np.seterr(over='ignore')

# 1. Parse Data
with open('/tmp/PMC3.md', 'r') as f:
    lines = f.readlines()

# Extract test data
test_lines = "".join(lines[104:150])
test_lines_nohtml = re.sub(r'<[^>]+>', ' ', test_lines)
test_nums = [float(x) for x in re.findall(r'(\d+\.\d{4})', test_lines_nohtml)]
d_test_all = np.array(test_nums)  # Shape (20,) for t=101..120

# Extract train data (Anexo)
train_lines = "".join(lines[183:])
train_lines_nohtml = re.sub(r'<[^>]+>', ' ', train_lines)
train_nums = [float(x) for x in re.findall(r'(\d+\.\d{4})', train_lines_nohtml)]

# Build correct series for t=1..100
# The table is arranged in 4 columns of t, f(t).
# Row 1: t=1, t=26, t=51, t=76
# Row 2: t=2, t=27, t=52, t=77
# ...
f_series = np.zeros(120)

idx = 0
for row in range(25):
    for col in range(4):
        t = row + 1 + col * 25
        f_series[t - 1] = train_nums[idx]
        idx += 1

# Add test data to the series
f_series[100:120] = d_test_all

def create_dataset(f_series, p, start_t, end_t):
    X = []
    d = []
    for t in range(start_t, end_t + 1):
        # t is 1-indexed. Array is 0-indexed.
        # To predict f(t), we need f(t-p) ... f(t-1)
        # In array index: f_series[t-1-p : t-1]
        x_in = f_series[t-1-p : t-1]
        X.append(x_in)
        d.append(f_series[t-1])
    return np.array(X), np.array(d).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W_hidden = np.random.rand(input_size, hidden_size)
        self.b_hidden = np.random.rand(1, hidden_size)
        
        self.W_output = np.random.rand(hidden_size, output_size)
        self.b_output = np.random.rand(1, output_size)

    def forward(self, X):
        self.net_hidden = np.dot(X, self.W_hidden) + self.b_hidden
        self.y_hidden = sigmoid(self.net_hidden)
        
        self.net_output = np.dot(self.y_hidden, self.W_output) + self.b_output
        self.y_output = sigmoid(self.net_output)
        return self.y_output
    
    def train(self, X, d, eta=0.1, alpha=0.8, max_epochs=100000, epsilon=0.5e-6):
        eqm_history = []
        prev_eqm = float('inf')
        
        v_W_hidden = np.zeros_like(self.W_hidden)
        v_b_hidden = np.zeros_like(self.b_hidden)
        v_W_output = np.zeros_like(self.W_output)
        v_b_output = np.zeros_like(self.b_output)
        
        for epoch in range(max_epochs):
            for i in range(X.shape[0]):
                xi = X[i:i+1] # 1 x inputs
                di = d[i:i+1] # 1 x outputs
                
                # Forward
                net_h = np.dot(xi, self.W_hidden) + self.b_hidden
                y_h = sigmoid(net_h)
                net_o = np.dot(y_h, self.W_output) + self.b_output
                y_o = sigmoid(net_o)
                
                # Backward
                error = di - y_o
                delta_o = error * d_sigmoid(y_o)
                
                delta_h = np.dot(delta_o, self.W_output.T) * d_sigmoid(y_h)
                
                # Deltas de pesos
                dW_output = eta * np.dot(y_h.T, delta_o) + alpha * v_W_output
                db_output = eta * delta_o + alpha * v_b_output
                
                dW_hidden = eta * np.dot(xi.T, delta_h) + alpha * v_W_hidden
                db_hidden = eta * delta_h + alpha * v_b_hidden
                
                # Atualização e guarda na velocidade (momentum)
                self.W_output += dW_output
                self.b_output += db_output
                v_W_output = dW_output
                v_b_output = db_output
                
                self.W_hidden += dW_hidden
                self.b_hidden += db_hidden
                v_W_hidden = dW_hidden
                v_b_hidden = db_hidden
            
            # Compute epoch EQM
            y_pred = self.forward(X)
            eqm = np.mean((d - y_pred)**2)
            eqm_history.append(eqm)
            
            if abs(prev_eqm - eqm) <= epsilon:
                break
                
            prev_eqm = eqm
            
        return eqm_history, epoch + 1


topologies = [
    {"name": "Rede 1", "p": 5, "N1": 10},
    {"name": "Rede 2", "p": 10, "N1": 15},
    {"name": "Rede 3", "p": 15, "N1": 25}
]

results = []

for topo in topologies:
    print(f"Training {topo['name']}...")
    X_train, d_train = create_dataset(f_series, topo['p'], topo['p'] + 1, 100)
    X_test, d_test = create_dataset(f_series, topo['p'], 101, 120)
    
    topo_res = []
    for t in range(3):
        seed = 42 + topo['p'] * 10 + t
        model = MLP(topo['p'], topo['N1'], 1, seed=seed)
        eqm_hist, epochs = model.train(X_train, d_train, eta=0.1, alpha=0.8, epsilon=0.5e-6)
        
        y_test_pred = model.forward(X_test)
        
        # Erro relativo médio
        # e = |d - y| / d
        # some d could be 0, but d_test values in table are strictly > 0
        err_rel = np.abs((d_test - y_test_pred) / d_test) * 100
        erm = np.mean(err_rel)
        variancia = np.var(err_rel)
        
        topo_res.append({
            "treinamento": t + 1,
            "eqm_final": eqm_hist[-1],
            "epocas": epochs,
            "eqm_history": eqm_hist,
            "y_test_pred": y_test_pred.flatten(),
            "erm": erm,
            "variancia": variancia,
            "model": model
        })
        print(f"  T{t+1}: Epocas={epochs}, EQM={eqm_hist[-1]:.6f}, ERM={erm:.2f}%")
        
    results.append({
        "topo": topo,
        "runs": topo_res
    })

# Encontrar o melhor treinamento para cada topologia (menor ERM ou menor EQM?)
# "considerando ainda o melhor treinamento realizado em cada uma delas, trace o gráfico..."
# Vou escolher o melhor baseado no menor Erro Relativo Médio no Teste (melhor generalização).
for r in results:
    best_run = min(r["runs"], key=lambda x: x["erm"])
    r["best_run"] = best_run

# 4. Gráficos de EQM para o melhor treinamento de cada topologia
plt.figure(figsize=(15, 5))
for i, r in enumerate(results):
    plt.subplot(1, 3, i+1)
    best = r["best_run"]
    plt.plot(best["eqm_history"], color='blue')
    plt.title(f"{r['topo']['name']} (T{best['treinamento']}) - Épocas: {best['epocas']}")
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.grid(True)
plt.tight_layout()
plt.savefig("graficos_eqm.png")

# 5. Gráfico Valores Desejados x Estimados para o melhor treinamento
plt.figure(figsize=(15, 5))
t_axis = np.arange(101, 121)
for i, r in enumerate(results):
    plt.subplot(1, 3, i+1)
    best = r["best_run"]
    plt.plot(t_axis, d_test_all, 'o-', label="Desejado", color='black')
    plt.plot(t_axis, best["y_test_pred"], 'x--', label="Estimado", color='red')
    plt.title(f"{r['topo']['name']} (T{best['treinamento']})")
    plt.xlabel("Tempo (t)")
    plt.ylabel("f(t)")
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig("graficos_estimados.png")

# Find the absolute best among all
best_overall = min(results, key=lambda x: x["best_run"]["erm"])

# HTML Generation
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Respostas PMC3</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        h1, h2 {{ color: #333; }}
        .question {{ font-weight: bold; margin-top: 30px; }}
        .answer {{ margin-bottom: 20px; text-align: justify; }}
    </style>
</head>
<body>
    <h1>Trabalho - Perceptron Multicamadas (Backpropagation) - PMC3</h1>
    
    <div class="question">2. Registre os resultados finais desses 3 treinamentos para cada uma das três topologias de rede na tabela a seguir:</div>
    <table>
        <tr>
            <th>Treinamento</th>
            <th colspan="2">Rede 1</th>
            <th colspan="2">Rede 2</th>
            <th colspan="2">Rede 3</th>
        </tr>
        <tr>
            <th></th>
            <th>EQM</th>
            <th>Épocas</th>
            <th>EQM</th>
            <th>Épocas</th>
            <th>EQM</th>
            <th>Épocas</th>
        </tr>
"""
for t in range(3):
    html += f"<tr><td>{t+1}º (T{t+1})</td>"
    for r in results:
        run = r["runs"][t]
        html += f"<td>{run['eqm_final']:.6f}</td><td>{run['epocas']}</td>"
    html += "</tr>"
html += "</table>"

html += """
    <div class="question">3. Faça a validação da rede em relação aos valores desejados apresentados na tabela abaixo.</div>
    <table>
        <tr>
            <th></th>
            <th></th>
            <th colspan="3">Rede 1</th>
            <th colspan="3">Rede 2</th>
            <th colspan="3">Rede 3</th>
        </tr>
        <tr>
            <th>Amostra</th>
            <th>f(t)</th>
            <th>(T1)</th><th>(T2)</th><th>(T3)</th>
            <th>(T1)</th><th>(T2)</th><th>(T3)</th>
            <th>(T1)</th><th>(T2)</th><th>(T3)</th>
        </tr>
"""

for i in range(20):
    html += f"<tr><td>t = {101+i}</td><td>{d_test_all[i]:.4f}</td>"
    for r in results:
        for t in range(3):
            val = r["runs"][t]["y_test_pred"][i]
            html += f"<td>{val:.4f}</td>"
    html += "</tr>"

html += "<tr><td><b>Erro Relativo Médio:</b></td><td></td>"
for r in results:
    for t in range(3):
        html += f"<td><b>{r['runs'][t]['erm']:.2f}%</b></td>"
html += "</tr><tr><td><b>Variância:</b></td><td></td>"
for r in results:
    for t in range(3):
        html += f"<td><b>{r['runs'][t]['variancia']:.2f}%</b></td>"
html += "</tr></table>"

html += """
    <div class="question">4. Trace o gráfico dos valores de erro quadrático médio (EQM) em função de cada época de treinamento para o melhor de cada rede.</div>
    <div class="answer">
        <p>Abaixo estão os gráficos do EQM para a melhor execução (menor Erro Relativo no Teste) de cada uma das três topologias:</p>
        <img src="graficos_eqm.png" alt="Gráficos de EQM" width="1000">
    </div>

    <div class="question">5. Trace o gráfico dos valores desejados e dos valores estimados pela respectiva rede em função do domínio de estimação.</div>
    <div class="answer">
        <p>Abaixo estão os gráficos comparando o valor desejado f(t) e o valor estimado y(t) para o conjunto de testes (t=101..120) para o melhor treinamento de cada topologia:</p>
        <img src="graficos_estimados.png" alt="Gráficos Valores Estimados" width="1000">
    </div>

    <div class="question">6. Indique qual das topologias candidatas e configuração final seria a mais adequada para previsão.</div>
    <div class="answer">
"""
html += f"Baseado nas análises das tabelas e gráficos, a topologia mais adequada é a <b>{best_overall['topo']['name']}</b> utilizando a configuração de treinamento <b>T{best_overall['best_run']['treinamento']}</b>. "
html += f"Esta combinação apresentou o menor Erro Relativo Médio ({best_overall['best_run']['erm']:.2f}%) sobre o conjunto de dados de teste (dados não vistos durante o treinamento), demonstrando a melhor capacidade de generalização e aderência aos dados temporais futuros (t=101 a 120)."
html += """
    </div>

    <div class="question">7. Investigue e comente sobre as principais características e vantagens dos seguintes algoritmos de treinamento:</div>
    <div class="answer">
        <b>a. Algoritmo de treinamento Resilient-Propagation (RProp)</b><br>
        O RProp (Resilient Backpropagation) é um algoritmo de aprendizado heurístico cuja principal característica é utilizar apenas o <b>sinal</b> (direção) do gradiente de erro para atualizar os pesos, ignorando a magnitude (valor absoluto) do gradiente. Para cada peso, ele mantém um fator de atualização individual. Se o gradiente mantiver o mesmo sinal por duas épocas seguidas, o tamanho do passo aumenta (aceleração); se o sinal inverter (indicando que o mínimo foi ultrapassado), o tamanho do passo diminui. <br>
        <i>Vantagens:</i> É extremamente rápido e robusto. Resolve problemas clássicos do gradiente descendente convencional, como a estagnação em regiões de platô (onde o gradiente é muito pequeno), permitindo uma convergência muito mais rápida. Não requer ajuste manual criterioso da taxa de aprendizagem para cada problema.
        <br><br>
        <b>b. Algoritmo de treinamento Levenberg-Marquardt (LM)</b><br>
        O algoritmo de Levenberg-Marquardt é uma técnica de otimização numérica que combina a velocidade do método de Newton (ou Gauss-Newton) com a estabilidade do método do Gradiente Descendente. Ele utiliza a matriz Jacobiana (derivadas dos erros em relação aos pesos) para aproximar a matriz Hessiana. Quando o algoritmo está longe do mínimo, ele se comporta como o gradiente descendente (passos pequenos e seguros); quando se aproxima do mínimo, ele transiciona para o método de Gauss-Newton, garantindo uma convergência quadrática e extremamente rápida.<br>
        <i>Vantagens:</i> É considerado um dos algoritmos de treinamento mais rápidos disponíveis para redes neurais de tamanho moderado. Apresenta excelente taxa de convergência e costuma encontrar erros muito menores que o backpropagation padrão em consideravelmente menos épocas. A principal desvantagem é o alto custo computacional por época e o grande consumo de memória, pois exige o cálculo e armazenamento da matriz Jacobiana, tornando-o inadequado para redes gigantescas.
    </div>
</body>
</html>
"""

with open('respostas.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML generated: respostas.html")
