import re
import time
import numpy as np
import matplotlib.pyplot as plt

np.seterr(over='ignore')

# 1. Parse Data
with open('/tmp/PMC2.md', 'r') as f:
    lines = f.readlines()

test_lines = "".join(lines[157:445])
test_lines_nohtml = re.sub(r'<[^>]+>', ' ', test_lines)
test_nums = [float(x) for x in test_lines_nohtml.split() if re.match(r'^\d+(\.\d+)?$', x)]

parsed_test = []
i = 0
while i < len(test_nums):
    if test_nums[i] == float(len(parsed_test) + 1):
        parsed_test.append(test_nums[i+1:i+8])
        i += 8
    else:
        i += 1

train_lines = "".join(lines[491:])
train_lines_nohtml = re.sub(r'<[^>]+>', ' ', train_lines)
train_nums = [float(x) for x in train_lines_nohtml.split() if re.match(r'^\d+(\.\d+)?$', x)]

parsed_train = []
amostras_found = set()
i = 0
while i < len(train_nums) - 7:
    amostra = int(train_nums[i])
    if amostra > 0 and amostra <= 300 and train_nums[i+1] <= 1.0 and train_nums[i+2] <= 1.0:
        if amostra not in amostras_found:
            amostras_found.add(amostra)
            parsed_train.append(train_nums[i+1:i+8])
    i += 1

X_train = np.array([x[:4] for x in parsed_train])
d_train = np.array([x[4:] for x in parsed_train])

X_test = np.array([x[:4] for x in parsed_test])
d_test = np.array([x[4:] for x in parsed_test])

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
        
    def copy_weights_from(self, other):
        self.W_hidden = other.W_hidden.copy()
        self.b_hidden = other.b_hidden.copy()
        self.W_output = other.W_output.copy()
        self.b_output = other.b_output.copy()

    def forward(self, X):
        self.net_hidden = np.dot(X, self.W_hidden) + self.b_hidden
        self.y_hidden = sigmoid(self.net_hidden)
        
        self.net_output = np.dot(self.y_hidden, self.W_output) + self.b_output
        self.y_output = sigmoid(self.net_output)
        return self.y_output
    
    def train(self, X, d, eta=0.1, alpha=0.0, max_epochs=100000, epsilon=1e-6):
        eqm_history = []
        prev_eqm = float('inf')
        
        # Para o momentum
        v_W_hidden = np.zeros_like(self.W_hidden)
        v_b_hidden = np.zeros_like(self.b_hidden)
        v_W_output = np.zeros_like(self.W_output)
        v_b_output = np.zeros_like(self.b_output)
        
        start_time = time.time()
        
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
            # EQM is mean squared error across all outputs and all samples
            eqm = np.mean((d - y_pred)**2)
            eqm_history.append(eqm)
            
            if abs(prev_eqm - eqm) <= epsilon:
                break
                
            prev_eqm = eqm
            
        elapsed = time.time() - start_time
        return eqm_history, epoch + 1, elapsed

print("Initializing models...")
seed = 42
model_padrao = MLP(4, 15, 3, seed)
model_momentum = MLP(4, 15, 3)
model_momentum.copy_weights_from(model_padrao)

print("Training standard backpropagation...")
eqm_padrao, epocas_padrao, time_padrao = model_padrao.train(X_train, d_train, eta=0.1, alpha=0.0, epsilon=1e-6)

print("Training backpropagation with momentum...")
eqm_momentum, epocas_momentum, time_momentum = model_momentum.train(X_train, d_train, eta=0.1, alpha=0.9, epsilon=1e-6)

# Generate plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(eqm_padrao, color='blue')
plt.title(f"Padrão (Épocas: {epocas_padrao}, Tempo: {time_padrao:.2f}s)")
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(eqm_momentum, color='red')
plt.title(f"Com Momentum (Épocas: {epocas_momentum}, Tempo: {time_momentum:.2f}s)")
plt.xlabel('Épocas')
plt.ylabel('EQM')
plt.grid(True)

plt.tight_layout()
plt.savefig('grafico_eqm.png')
print("Graph saved.")

# Testing
def test_model(model, X, d):
    y_pred = model.forward(X)
    # Rounding
    y_round = np.where(y_pred >= 0.5, 1, 0)
    
    # Calculate accuracy
    # A sample is correct only if all 3 output neurons match
    matches = np.all(y_round == d, axis=1)
    acc = np.mean(matches) * 100
    
    return y_pred, y_round, acc

print("Testing standard model...")
y_pred_padrao, y_round_padrao, acc_padrao = test_model(model_padrao, X_test, d_test)

print("Testing momentum model...")
y_pred_momentum, y_round_momentum, acc_momentum = test_model(model_momentum, X_test, d_test)

# HTML Generation
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Respostas PMC2</title>
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
    <h1>Trabalho - Perceptron Multicamadas (Backpropagation) - PMC2</h1>
    
    <div class="question">1. e 2. Execute os treinamentos (Padrão e com Momentum) e trace os gráficos</div>
    <div class="answer">
        <p>Abaixo seguem os resultados das duas execuções solicitadas (padrão e com momentum):</p>
        <ul>
            <li><b>Backpropagation Padrão (η=0.1, sem momentum):</b> {epocas_padrao} épocas, Tempo: {time_padrao:.2f} segundos.</li>
            <li><b>Backpropagation com Momentum (η=0.1, α=0.9):</b> {epocas_momentum} épocas, Tempo: {time_momentum:.2f} segundos.</li>
        </ul>
        <p>Os gráficos do Erro Quadrático Médio em função das épocas para ambos os treinamentos estão na imagem abaixo (grafico_eqm.png):</p>
        <img src="grafico_eqm.png" alt="Gráficos de EQM" width="800">
    </div>
    
    <div class="question">3. e 4. Faça a validação da rede aplicando o conjunto de teste fornecido... Forneça a taxa de acerto (%).</div>
    <div class="answer">
        A tabela a seguir mostra os resultados da rede treinada com <b>Backpropagation Padrão</b> (já que o objetivo principal é classificação, e ambos os treinamentos usam os mesmos dados, utilizaremos o modelo padrão aqui como exemplo representativo para a tabela. A taxa de acerto do modelo com momentum também foi calculada e é de {acc_momentum:.2f}%).
    </div>
    
    <table>
        <tr>
            <th>Amostra</th>
            <th>x1</th>
            <th>x2</th>
            <th>x3</th>
            <th>x4</th>
            <th>d1</th>
            <th>d2</th>
            <th>d3</th>
            <th>y1</th>
            <th>y2</th>
            <th>y3</th>
            <th>y1 (arred.)</th>
            <th>y2 (arred.)</th>
            <th>y3 (arred.)</th>
        </tr>
"""

for i in range(len(X_test)):
    html += "<tr>"
    html += f"<td>{i+1}</td>"
    html += f"<td>{X_test[i][0]:.4f}</td>"
    html += f"<td>{X_test[i][1]:.4f}</td>"
    html += f"<td>{X_test[i][2]:.4f}</td>"
    html += f"<td>{X_test[i][3]:.4f}</td>"
    html += f"<td>{int(d_test[i][0])}</td>"
    html += f"<td>{int(d_test[i][1])}</td>"
    html += f"<td>{int(d_test[i][2])}</td>"
    
    html += f"<td>{y_pred_padrao[i][0]:.4f}</td>"
    html += f"<td>{y_pred_padrao[i][1]:.4f}</td>"
    html += f"<td>{y_pred_padrao[i][2]:.4f}</td>"
    
    html += f"<td><b>{int(y_round_padrao[i][0])}</b></td>"
    html += f"<td><b>{int(y_round_padrao[i][1])}</b></td>"
    html += f"<td><b>{int(y_round_padrao[i][2])}</b></td>"
    html += "</tr>"

html += f"""
        <tr>
            <td colspan="11" style="text-align: right;"><b>Taxa de Acerto (%):</b></td>
            <td colspan="3"><b>{acc_padrao:.2f}%</b></td>
        </tr>
    </table>
    
</body>
</html>
"""

with open('respostas.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("HTML generated: respostas.html")
