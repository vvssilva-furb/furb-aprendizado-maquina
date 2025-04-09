import pandas as pd
import numpy as np
import demo as reg_simples
import matplotlib.pyplot as plt

def regressaop():
    """
    a) Baixe o arquivo data_preg.mat ou data_preg.csv. A primeira coluna representa os valores de x e a segunda coluna representa os valores de y.

    """
    data = pd.read_csv("data_preg.csv", header=None)
    vetor_x = data.iloc[:, 0]
    vetor_y = data.iloc[:, 1]

    """
    b) Faça o Gráfico de dispersão dos dados.
    """
    correlacao = reg_simples.correlacao(vetor_x, vetor_y)
    reg_simples.gera_graficos(vetor_x, vetor_y)

    """
    c) Use a função polyfit para gerar a linha de regressão para N = 1 e trace-o no gráfico de dispersão na cor vermelha (plot (x, y, 'r')). 
    (Observe que nesta função a numeração coeficiente é invertida! B0=BN , B1=BN−1 , B2=BN−2 , ...BN=B0)
    """
    regressao_n1 = np.polyfit(vetor_x, vetor_y, 1)
    ## inverte valor resultante do polyfit
    regressao_n1 = regressao_n1[::-1]
    #print(regressao_n1)

    gera_graficos(vetor_x, vetor_y, correlacao, regressao_n1, "r")

    """
    d) Trace a linha de regressão para N = 2 no gráfico na cor verde. 
    Para isso, você deverá calcular esta função y = 𝛽0 + 𝛽1X + 𝛽2X2 + 𝛽3X3 + …+ 𝛽 NXN, isto é, não pode usar a função pronta do python.
    """

    # Construir a matriz do sistema (polinômio de grau 2)
    X_N2 = np.vstack([np.ones_like(vetor_x), vetor_x, vetor_x ** 2]).T  # Cada linha: [1, x, x^2]
    #X_N2 é transposta para organizar os dados corretamente ^^^
    Y_N2 = vetor_y  # Vetor de respostas

    # Resolver a equação normal B = (A^T A)^(-1) A^T Y
    regressao_n2 = np.linalg.inv(X_N2.T @ X_N2) @ X_N2.T @ Y_N2  # Obtém os coeficientes [B0, B1, B2]
    regressao_n2 = regressao_n2[::-1]
    print(regressao_n2)
    print(np.polyfit(vetor_x, vetor_y, 2))
    gera_graficos(vetor_x, vetor_y, correlacao, regressao_n2, "g")

    """
    e) Trace a linha de regressão para N = 3 no gráfico na cor preta.
    """

    # Construir a matriz do sistema (polinômio de grau 3)
    X_N3 = np.vstack([np.ones_like(vetor_x), vetor_x, vetor_x ** 2, vetor_x ** 3]).T  # Cada linha: [1, x, x^2]
    Y_N3 = vetor_y  # Vetor de respostas

    # Resolver a equação normal B = (A^T A)^(-1) A^T Y
    regressao_n3 = np.linalg.inv(X_N3.T @ X_N3) @ X_N3.T @ Y_N3  # Obtém os coeficientes [B0, B1, B2, ...]
    regressao_n3 = regressao_n3[::-1]
    print(regressao_n3)
    print(np.polyfit(vetor_x, vetor_y, 3))
    gera_graficos(vetor_x, vetor_y, correlacao, regressao_n3, "black")

    """
    f) Trace a linha de regressão para N = 8 no gráfico na cor amarela.
    """
    # Construir a matriz do sistema (polinômio de grau 8)
    X_N8 = np.vstack([np.ones_like(vetor_x), vetor_x, vetor_x ** 2, vetor_x ** 3,
                   vetor_x ** 4, vetor_x ** 5, vetor_x ** 6, vetor_x ** 7, vetor_x ** 8]).T  # Cada linha: [1, x, x^2]
    Y_N8 = vetor_y  # Vetor de respostas

    # Resolver a equação normal B = (A^T A)^(-1) A^T Y
    regressao_n8 = np.linalg.inv(X_N8.T @ X_N8) @ X_N8.T @ Y_N8  # Obtém os coeficientes [B0, B1, B2, ...]
    regressao_n8 = regressao_n8[::-1]
    print(regressao_n8)
    print(np.polyfit(vetor_x, vetor_y, 8))
    gera_graficos(vetor_x, vetor_y, correlacao, regressao_n8, "yellow")

    """
    g) Calcule o Erro Quadrático Médio (EQM) para cada linha de regressão. Qual é o mais preciso?
    """
    ##Regressão N=1
    X_N1 = np.vstack([np.ones_like(vetor_x), vetor_x]).T  # Cada linha: [1, x, x^2]

    EQM_numpy_n1 = eqm_numpy(X_N1, regressao_n1, vetor_y)
    print("Erro Quadrático Médio N=1 (EQM no Numpy):", EQM_numpy_n1)
    EQM_n1 = eqm(regressao_n1, vetor_x, vetor_y)
    print("Erro Quadrático Médio N=1 (EQM):", EQM_n1)

    ##Regressão N=2
    EQM_numpy_n2 = eqm_numpy(X_N2, regressao_n2, vetor_y)
    print("Erro Quadrático Médio N=2 (EQM no Numpy):", EQM_numpy_n2)
    EQM_n2 = eqm(regressao_n2, vetor_x, vetor_y)
    print("Erro Quadrático Médio N=2 (EQM):", EQM_n2)

    ##Regressão N=3
    EQM_numpy_n3 = eqm_numpy(X_N3, regressao_n3, vetor_y)
    print("Erro Quadrático Médio N=3 (EQM no Numpy):", EQM_numpy_n3)
    EQM_n3 = eqm(regressao_n3, vetor_x, vetor_y)
    print("Erro Quadrático Médio N=3 (EQM):", EQM_n3)

    ##Regressão N=8
    EQM_numpy_n8 = eqm_numpy(X_N8, regressao_n8, vetor_y)
    print("Erro Quadrático Médio N=8 (EQM no Numpy):", EQM_numpy_n8)
    EQM_n8 = eqm(regressao_n8, vetor_x, vetor_y)
    print("Erro Quadrático Médio N=8 (EQM):", EQM_n8)

    """
    h) Para evitar o overfitting, divida os dados aleatoriamente em Dados de Treinamento e Dados de Teste. 
    Use 10% dos dados como conjunto de teste, e o resto como de treinamento.
    
    OBS: adicionamos comentários com o passo-a-passo, seguindo os slides do conteúdo (Método do Test Set)
    """

    # Passo 1: Embaralhar os índices aleatoriamente
    total_dados = len(vetor_x)
    indices = list(range(total_dados))
    np.random.seed(42)  # para reprodução dos resultados
    np.random.shuffle(indices)

    # Passo 2: Separar 10% para teste e 90% para treinamento
    qtd_teste = int(0.1 * total_dados)
    indices_teste = indices[:qtd_teste]
    indices_treinamento = indices[qtd_teste:]

    # Criar conjuntos de treino e teste
    x_treino = vetor_x.iloc[indices_treinamento].values
    y_treino = vetor_y.iloc[indices_treinamento].values

    x_teste = vetor_x.iloc[indices_teste].values
    y_teste = vetor_y.iloc[indices_teste].values

    # Passo 3: Calcular a Regressão Linear (N = 1) no grupo de treinamento
    # y = beta_0 + beta_1 * x
    X_treino_n1 = np.vstack([np.ones_like(x_treino), x_treino]).T
    Y_treino = y_treino

    coef_n1_treino = np.linalg.inv(X_treino_n1.T @ X_treino_n1) @ X_treino_n1.T @ Y_treino
    coef_n1_treino = coef_n1_treino[::-1]

    # Passo 4: Estimar os valores de y no conjunto de teste com os coeficientes encontrados
    X_teste_n1 = np.vstack([np.ones_like(x_teste), x_teste]).T
    y_estimado_teste = X_teste_n1 @ coef_n1_treino

    # Passo 5: Calcular o EQM nos dados de teste
    eqm_teste = np.mean((y_teste - y_estimado_teste) ** 2)

    print("\n--- Validação com divisão 90/10 (manual, sem bibliotecas externas) ---")
    print("Coeficientes da regressão N=1 (treinamento):", coef_n1_treino)
    print("EQM no conjunto de teste (10% dos dados):", eqm_teste)

    """
    i) Repita os passos de c - f, mas agora use apenas os dados de treinamento para ajustar a linha de regressão.
    """
    print("\n--- Validação com dados de TREINAMENTO e teste separados ---")

    for grau in [1, 2, 3, 8]:
        # Construir a matriz de treinamento para o grau atual
        X_treino = np.vstack([x_treino ** i for i in range(grau + 1)]).T
        coef = np.linalg.inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
        coef = coef[::-1]  # inverter para manter ordem como no restante do código

        # Construir matriz de teste para o grau atual
        X_teste = np.vstack([x_teste ** i for i in range(grau + 1)]).T
        y_teste_estimado = X_teste @ coef

        # Calcular EQM no conjunto de teste
        eqm_val = np.mean((y_teste - y_teste_estimado) ** 2)

        # Mostrar resultados
        print(f"Grau {grau}: Coeficientes = {coef}")
        print(f"Grau {grau}: EQM no conjunto de teste = {eqm_val}\n")

def eqm(regressao_n, vetor_x, vetor_y):
    # 1. Calcular os valores estimados (ŷ) para qualquer N
    y_estimado = [sum(regressao_n[j] * (vetor_x[i] ** j) for j in range(len(regressao_n))) for i in range(len(vetor_x))]

    # 2. Calcular os resíduos ao quadrado
    residuo_quadrado = [(vetor_y[i] - y_estimado[i]) ** 2 for i in range(len(vetor_y))]

    # 3. Calcular o EQM
    EQM = sum(residuo_quadrado) / len(vetor_y)

    return EQM

"""
Criamos este método para fins de comparação, entre o numpy e a implementação que fizemos.
"""
def eqm_numpy(X, regressao_n, vetor_y):
    y_estimado = X @ regressao_n
    EQM_numpy = np.mean(np.square(vetor_y - y_estimado))

    return EQM_numpy

def gera_graficos(vetor_x, vetor_y, correlacao, regressao, cor_linha_regressao):
    menor_x = int(min(vetor_x)) - 1
    maior_x = int(max(vetor_x)) + 2
    array_x = []
    array_y = []

    for i in range(menor_x, maior_x):
        array_x.append(i)
        array_y.append(regressao[0] + (regressao[1] * i))

    betas_titulo =  ", ".join([f"beta {i} = {regressao[i]:.4f}" for i in range(len(regressao))])

    plt.scatter(vetor_x, vetor_y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(
        f"Gráfico de Dispersão \nRegressão: {betas_titulo} \nCorrelação: r = {correlacao:.4f}")
    plt.plot(array_x, array_y, color=cor_linha_regressao)
    plt.show()

if __name__ == '__main__':
    regressaop()