import re
import numpy as np
import matplotlib.pyplot as plt

np.seterr(over='ignore')

# 1. Parse Data
with open('/tmp/PMC1.md', 'r') as f:
    lines = f.readlines()

test_lines = "".join(lines[151:391])
test_lines_nohtml = re.sub(r'<[^>]+>', ' ', test_lines)
test_nums = [float(x) for x in test_lines_nohtml.split() if re.match(r'^\d+(\.\d+)?$', x)]

parsed_test = []
i = 0
while i < len(test_nums):
    if test_nums[i] == float(len(parsed_test) + 1):
        parsed_test.append([test_nums[i+1], test_nums[i+2], test_nums[i+3], test_nums[i+4]])
        i += 5
    else:
        i += 1

train_lines = "".join(lines[457:])
train_lines_nohtml = re.sub(r'<[^>]+>', ' ', train_lines)
train_nums = [float(x) for x in train_lines_nohtml.split() if re.match(r'^\d+(\.\d+)?$', x)]

parsed_train = []
amostras_found = set()
i = 0
while i < len(train_nums) - 4:
    amostra = int(train_nums[i])
    if amostra > 0 and amostra <= 300 and train_nums[i+1] <= 1.0 and train_nums[i+2] <= 1.0:
        if amostra not in amostras_found:
            amostras_found.add(amostra)
            parsed_train.append([train_nums[i+1], train_nums[i+2], train_nums[i+3], train_nums[i+4]])
    i += 1

X_train = np.array([x[:3] for x in parsed_train])
d_train = np.array([x[3] for x in parsed_train]).reshape(-1, 1)

X_test = np.array([x[:3] for x in parsed_test])
d_test = np.array([x[3] for x in parsed_test]).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y):
    return y * (1 - y)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed):
        np.random.seed(seed)
        # Inicializando com valores aleatórios entre 0 e 1
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
    
    def train(self, X, d, eta=0.1, max_epochs=100000, epsilon=1e-6):
        eqm_history = []
        prev_eqm = float('inf')
        
        for epoch in range(max_epochs):
            # Online learning (stochastic gradient descent)
            for i in range(X.shape[0]):
                xi = X[i:i+1] # 1x3
                di = d[i:i+1] # 1x1
                
                # Forward
                net_h = np.dot(xi, self.W_hidden) + self.b_hidden
                y_h = sigmoid(net_h)
                net_o = np.dot(y_h, self.W_output) + self.b_output
                y_o = sigmoid(net_o)
                
                # Backward
                error = di - y_o
                delta_o = error * d_sigmoid(y_o)
                
                delta_h = np.dot(delta_o, self.W_output.T) * d_sigmoid(y_h)
                
                # Update weights
                self.W_output += eta * np.dot(y_h.T, delta_o)
                self.b_output += eta * delta_o
                
                self.W_hidden += eta * np.dot(xi.T, delta_h)
                self.b_hidden += eta * delta_h
            
            # Compute epoch EQM
            y_pred = self.forward(X)
            eqm = np.mean((d - y_pred)**2)
            eqm_history.append(eqm)
            
            if abs(prev_eqm - eqm) <= epsilon:
                break
                
            prev_eqm = eqm
            
        return eqm_history, epoch + 1

# Execute 5 trainings
results = []
models = []

print("Starting trainings...")
for t in range(5):
    # Try different seeds just to ensure different initializations
    seed = t * 100 + 42
    model = MLP(3, 10, 1, seed)
    eqm_hist, epochs = model.train(X_train, d_train, eta=0.1, epsilon=1e-6)
    results.append({
        'treinamento': t+1,
        'eqm_final': eqm_hist[-1],
        'epocas': epochs,
        'eqm_history': eqm_hist
    })
    models.append(model)
    print(f"T{t+1}: Epocas={epochs}, EQM={eqm_hist[-1]:.6f}")

# Sort to find 2 longest trainings
longest_trainings = sorted(results, key=lambda x: x['epocas'], reverse=True)[:2]

plt.figure(figsize=(10, 5))
for i, lt in enumerate(longest_trainings):
    plt.subplot(1, 2, i+1)
    plt.plot(lt['eqm_history'])
    plt.title(f"Treinamento {lt['treinamento']} (Épocas: {lt['epocas']})")
    plt.xlabel('Épocas')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_eqm.png')
print("Graph saved to grafico_eqm.png")

# Calculate Test Predictions
test_table = []
test_metrics = []

for m_idx, model in enumerate(models):
    y_test_pred = model.forward(X_test)
    
    # Error relativo = |(d - y) / d| * 100
    errors = np.abs((d_test - y_test_pred) / d_test) * 100
    mean_error = np.mean(errors)
    variance = np.var(errors)
    
    test_metrics.append({
        'mean_error': mean_error,
        'variance': variance,
        'preds': y_test_pred.flatten()
    })

# HTML Generation
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Respostas PMC1</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; }
        h1, h2 { color: #333; }
        .question { font-weight: bold; margin-top: 30px; }
        .answer { margin-bottom: 20px; text-align: justify; }
    </style>
</head>
<body>
    <h1>Trabalho - Perceptron Multicamadas (Backpropagation)</h1>
    
    <div class="question">1. Execute 5 treinamentos...</div>
    <div class="question">2. Registre os resultados finais desses 5 treinamentos na tabela abaixo:</div>
    <table>
        <tr>
            <th>Treinamento</th>
            <th>Erro Quadrático Médio</th>
            <th>Número de Épocas</th>
        </tr>
"""
for r in results:
    html += f"<tr><td>{r['treinamento']}º (T{r['treinamento']})</td><td>{r['eqm_final']:.6f}</td><td>{r['epocas']}</td></tr>"

html += """
    </table>
    
    <div class="question">3. Para os dois treinamentos acima com maiores números de épocas, trace os respectivos gráficos...</div>
    <div class="answer">
        <p>Os gráficos do Erro Quadrático Médio em função das épocas estão na imagem abaixo (grafico_eqm.png):</p>
        <img src="grafico_eqm.png" alt="Gráficos de EQM" width="800">
    </div>
    
    <div class="question">4. Baseado na tabela do item 2, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento.</div>
    <div class="answer">
        A variação no Erro Quadrático Médio (EQM) final e no número de épocas entre diferentes treinamentos se deve principalmente à inicialização aleatória dos pesos sinápticos e dos vieses (bias) no início de cada treinamento. A superfície de erro da rede neural (que é percorrida pelo algoritmo de otimização de gradiente descendente) é altamente não linear e pode conter vários mínimos locais, vales e regiões planas (platôs). Dependendo do ponto de partida (determinado pelos pesos iniciais aleatórios), o algoritmo backpropagation pode seguir diferentes caminhos de descida. Isso faz com que a rede demore mais ou menos épocas para alcançar o critério de parada (precisão de 10<sup>-6</sup>). Além disso, a convergência pode ocorrer em mínimos locais diferentes que possuem valores de EQM final distintos.
    </div>
    
    <div class="question">5. Para todos os treinamentos efetuados no item 2, faça a validação da rede aplicando o conjunto de teste...</div>
    <table>
        <tr>
            <th>Amostra</th>
            <th>x1</th>
            <th>x2</th>
            <th>x3</th>
            <th>d</th>
            <th>y_rede (T1)</th>
            <th>y_rede (T2)</th>
            <th>y_rede (T3)</th>
            <th>y_rede (T4)</th>
            <th>y_rede (T5)</th>
        </tr>
"""

for i in range(20):
    html += f"<tr>"
    html += f"<td>{i+1}</td>"
    html += f"<td>{X_test[i][0]:.4f}</td>"
    html += f"<td>{X_test[i][1]:.4f}</td>"
    html += f"<td>{X_test[i][2]:.4f}</td>"
    html += f"<td>{d_test[i][0]:.4f}</td>"
    for m in range(5):
        html += f"<td>{test_metrics[m]['preds'][i]:.4f}</td>"
    html += f"</tr>"

html += f"<tr><td colspan='5'><b>Erro Relativo Médio (%)</b></td>"
for m in range(5):
    html += f"<td><b>{test_metrics[m]['mean_error']:.4f}%</b></td>"
html += "</tr>"

html += f"<tr><td colspan='5'><b>Variância (%)</b></td>"
for m in range(5):
    html += f"<td><b>{test_metrics[m]['variance']:.4f}%</b></td>"
html += "</tr>"

# Find best model
best_idx = np.argmin([tm['mean_error'] for tm in test_metrics])
best_T = best_idx + 1

html += f"""
    </table>
    
    <div class="question">6. Baseado nas análises da tabela acima indique qual das configurações finais de treinamento seria a mais adequada para o sistema...</div>
    <div class="answer">
        Baseado na tabela de testes, a configuração <b>T{best_T}</b> seria a mais adequada para o sistema de ressonância magnética. Essa configuração apresenta o menor Erro Relativo Médio e uma baixa variância ao ser aplicada ao conjunto de testes (dados inéditos que não fizeram parte do treinamento). Isso demonstra que essa configuração conseguiu a melhor generalização da função mapeada, ao invés de apenas decorar os padrões de treinamento.
    </div>
    
</body>
</html>
"""

with open('respostas.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML generated: respostas.html")
