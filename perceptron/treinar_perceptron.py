import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def func_ativacao(v):
    return 1.0 if v >= 0 else -1.0

def treinar_perceptron(X_train, d_train, taxa_aprendizagem=0.01, max_epocas=1000):
    num_amostras, num_features = X_train.shape
    
    # Conforme solicitado na atividade: inicializar com valores aleatórios entre 0 e 1
    w_inicial = np.random.uniform(0, 1, num_features)
    w = np.copy(w_inicial)
    
    epoca = 0
    erros = True
    
    # Lista para armazenar a quantidade de erros em cada época
    historico_erros = []
    
    while erros and epoca < max_epocas:
        erros = False
        erros_na_epoca = 0
        
        for i in range(num_amostras):
            x_i = X_train[i]
            d_i = d_train[i]
            
            # Campo local induzido
            v = np.dot(w, x_i)
            
            # Saída do perceptron
            y = func_ativacao(v)
            
            # Verificar se houve erro
            if y != d_i:
                # Regra de Hebb supervisionada
                w = w + taxa_aprendizagem * d_i * x_i
                erros = True
                erros_na_epoca += 1
                
        historico_erros.append(erros_na_epoca)
        epoca += 1
        
    return w_inicial, w, epoca, historico_erros

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    caminho_treino = os.path.join(dir_path, 'treinamento.csv')
    caminho_teste = os.path.join(dir_path, 'teste.csv')
    
    # Carregando os CSVs
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)
    
    # Montar matriz X de treino incorporando o bias (x0 = -1) como primeira coluna
    X_treino_raw = df_treino[['x1', 'x2', 'x3']].values
    num_amostras_treino = X_treino_raw.shape[0]
    X_treino = np.hstack((np.full((num_amostras_treino, 1), -1.0), X_treino_raw))
    d_treino = df_treino['d'].values
    
    # Montar matriz X de teste incorporando o bias (x0 = -1)
    X_teste_raw = df_teste[['x1', 'x2', 'x3']].values
    num_amostras_teste = X_teste_raw.shape[0]
    X_teste = np.hstack((np.full((num_amostras_teste, 1), -1.0), X_teste_raw))
    
    pesos_treinados = []
    pesos_iniciais_lista = []
    historico_total = []
    
    print("Iniciando os 5 processos de treinamento (pesos iniciais aleatórios)...")
    for t in range(1, 6):
        w_inicial, w, epocas, historico_erros = treinar_perceptron(X_treino, d_treino, taxa_aprendizagem=0.01, max_epocas=1000)
        pesos_treinados.append(w)
        pesos_iniciais_lista.append(w_inicial)
        historico_total.append(historico_erros)
        print(f"[T{t}] Convergiu em {epocas} épocas.")
        print(f"     Pesos iniciais (w0..w3): {np.round(w_inicial, 4)}")
        print(f"     Pesos finais   (w0..w3): {np.round(w, 4)}")
        
    # ---------------------------------------------------------
    # GERAR GRÁFICOS (ERROS vs ÉPOCAS)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Definindo cores e estilos para melhor visualização
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for t in range(5):
        # Eixo x = número da época (1 até len(historico)), Eixo y = quantidade de erros
        plt.plot(range(1, len(historico_total[t]) + 1), historico_total[t], 
                 label=f'Processo T{t+1}', color=cores[t], alpha=0.8, linewidth=2)
        
    plt.title('Curva de Aprendizado do Perceptron (5 Processos Independentes)', fontsize=14, pad=15)
    plt.xlabel('Número de Épocas', fontsize=12)
    plt.ylabel('Quantidade de Erros de Classificação', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper right')
    
    caminho_grafico = os.path.join(dir_path, 'erros_por_epoca.png')
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.close() # Libera a memória do pyplot
    print(f"\n[+] Gráfico salvo com sucesso em: {caminho_grafico}")
    
    # ---------------------------------------------------------
    # GERAR GRÁFICO 2 (DISPERSÃO 3D DOS DADOS E HIPERPLANO DE SEPARAÇÃO)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Separar os pontos do DataFrame por classe
    classe_positiva = df_treino[df_treino['d'] == 1]
    classe_negativa = df_treino[df_treino['d'] == -1]
    
    # Plotar as duas classes no espaço 3D
    ax.scatter(classe_positiva['x1'], classe_positiva['x2'], classe_positiva['x3'], 
               c='blue', marker='o', s=80, edgecolors='k', label='Classe C2 (+1)')
    ax.scatter(classe_negativa['x1'], classe_negativa['x2'], classe_negativa['x3'], 
               c='red', marker='^', s=80, edgecolors='k', label='Classe C1 (-1)')
               
    # Extrair os pesos do primeiro treinamento (T1) para plotar o hiperplano
    w_t1 = pesos_treinados[0]
    
    # Criar uma malha (grid) para x1 e x2 com base nos limites dos dados reais
    margin = 0.5
    x1_range = np.linspace(df_treino['x1'].min() - margin, df_treino['x1'].max() + margin, 10)
    x2_range = np.linspace(df_treino['x2'].min() - margin, df_treino['x2'].max() + margin, 10)
    xx, yy = np.meshgrid(x1_range, x2_range)
    
    # Calcular x3 do plano baseado nos pesos w_t1
    # Equação: v = w0*(-1) + w1*x1 + w2*x2 + w3*x3 = 0
    # Logo: x3 = (w0 - w1*x1 - w2*x2) / w3
    zz = (w_t1[0] - w_t1[1]*xx - w_t1[2]*yy) / w_t1[3]
    
    # Desenhar a superfície do plano de separação com transparência
    ax.plot_surface(xx, yy, zz, alpha=0.35, color='mediumseagreen', edgecolor='none')
    
    ax.set_title('Dispersão 3D das Amostras e Hiperplano de Separação (Modelo T1)', fontsize=15, pad=15)
    ax.set_xlabel('Grandeza x1', fontsize=12)
    ax.set_ylabel('Grandeza x2', fontsize=12)
    ax.set_zlabel('Grandeza x3', fontsize=12)
    
    # Ajustar o ângulo de visão para mostrar bem a separação
    ax.view_init(elev=15, azim=60)
    ax.legend(fontsize=11, loc='upper left')
    
    caminho_grafico_3d = os.path.join(dir_path, 'hiperplano_3d.png')
    plt.savefig(caminho_grafico_3d, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico 3D salvo com sucesso em: {caminho_grafico_3d}")
        
    print("\n-----------------------------------------------------------------------------------------")
    print(" Tabela de Resultados da Classificação das Amostras de Teste")
    print("-----------------------------------------------------------------------------------------")
    print("Amostra\tx1\t\tx2\t\tx3\t\ty(T1)\ty(T2)\ty(T3)\ty(T4)\ty(T5)")
    print("-----------------------------------------------------------------------------------------")
    
    for i in range(num_amostras_teste):
        amostra = int(df_teste['Amostra'].iloc[i])
        x1 = X_teste_raw[i][0]
        x2 = X_teste_raw[i][1]
        x3 = X_teste_raw[i][2]
        
        saidas = []
        for t in range(5):
            w = pesos_treinados[t]
            v = np.dot(w, X_teste[i])
            y = func_ativacao(v)
            saidas.append(int(y))
            
        print(f"{amostra:02d}\t{x1: .4f}\t{x2: .4f}\t{x3: .4f}\t{saidas[0]:>5}\t{saidas[1]:>5}\t{saidas[2]:>5}\t{saidas[3]:>5}\t{saidas[4]:>5}")
    print("-----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
