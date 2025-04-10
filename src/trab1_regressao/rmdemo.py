import demo as reg_simples
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli
"""

def regmultipla():
    data = pd.read_csv("data.csv", header=None)

    """
    c)       Gere uma matriz X para as variáveis independentes (que são o tamanho da casa e o número de quartos) 
    e o vetor y da variável dependente (que é o preço).
    """
    matriz_X = data.iloc[:, :2]
    vetor_y = data.iloc[:, 2]
    ##print(matriz_X)
    ##print(vetor_y)

    """
    d)      Verifique a correlação e a regressão para Tamanho da casa 
    e Preço, e, Número de quartos e Preço e apresente os valores no gráfico de dispersão.
    """
    vetor_tamanho_casa = data.iloc[:, 0]
    correlacao_tamanho_casa = reg_simples.correlacao(vetor_tamanho_casa, vetor_y)
    regressao_tamanho_casa = reg_simples.regressao(vetor_tamanho_casa, vetor_y)
    reg_simples.gera_graficos(vetor_tamanho_casa, vetor_y)
    print(regressao_tamanho_casa)

    vetor_numero_quartos = data.iloc[:, 1]
    correlacao_numero_quartos = reg_simples.correlacao(vetor_numero_quartos, vetor_y)
    regressao_numero_quartos = reg_simples.regressao(vetor_numero_quartos, vetor_y)
    reg_simples.gera_graficos(vetor_numero_quartos, vetor_y)
    print(regressao_numero_quartos)

    matriz_X.insert(0, " ", 1)
    beta = (np.linalg.inv(np.transpose(matriz_X.to_numpy()) @ matriz_X.to_numpy())
            @ np.matrix_transpose(matriz_X)
            @ vetor_y.to_numpy())
    print(beta)

    # Criando o gráfico
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotando os dados
    ax.scatter(matriz_X.iloc[:, 1], matriz_X.iloc[:, 2], vetor_y, color='b', marker='o')

    # Criando uma grade para o plano de regressão
    ponto_x = np.linspace(matriz_X.iloc[:, 1].to_numpy().min(), matriz_X.iloc[:, 1].to_numpy().max(), 10)
    ponto_y = np.linspace(matriz_X.iloc[:, 2].to_numpy().min(), matriz_X.iloc[:, 2].to_numpy().max(), 10)

    # Usar meshgrid para criar a grade de combinações de X1 e X2
    X1, X2 = np.meshgrid(ponto_x, ponto_y)

    # Criar a matriz X (adicionando a coluna de 1s para o intercepto)
    X = np.ones((X1.size, 3))  # Inicializa a matriz X com 1s, tamanho de X1.size para se ajustar -> X1.size define o número de linhas, e 3 define o número de colunas
    X[:, 1] = X1.flatten()  # Coloca X1 na segunda coluna -> transforma array 2D em 1D
    X[:, 2] = X2.flatten()  # Coloca X2 na terceira coluna

    # Calcular Y multiplicando X pela matriz beta
    Y = np.dot(X, beta) # -> produto escalar entre matrizes

    # Reshape de Y para a forma de X1, X2 (métrica para o gráfico)
    Y = Y.reshape(X1.shape)

    ax.plot_surface(X1, X2, Y)

    # Definindo os rótulos dos eixos
    ax.set_xlabel('Tamanho da Casa (m²)')
    ax.set_ylabel('Número de Quartos')
    ax.set_zlabel('Preço da Casa (R$)')

    # Exibindo o gráfico
    plt.title(
        f"Correlação - tamanho da casa e preço: r = {correlacao_tamanho_casa:.4f} \nCorrelação - n° de quartos e preço: r = {correlacao_numero_quartos:.4f}")
    #plt.show()

    questao_h_matriz_x = [[1, 1650, 3]]
    #questao_h_matriz_x = [[1, 2500, 4]]
    print(questao_h_matriz_x)

    questao_h_matriz_y =  np.dot(questao_h_matriz_x, beta)
    print(questao_h_matriz_y)

    """
    preço aumentou após diminuir qtd. de quartos
    Motivo: acreditamos que é devido a correlação entre n° de quartos e preço ser menor do que a correlação entre
        tamanho da casa e preço
    """

if __name__ == '__main__':
    regmultipla()