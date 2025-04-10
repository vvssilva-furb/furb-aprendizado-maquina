import scipy.io as scipy
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

"""
Trabalho 2: Classificação - k-Nearest Neighbour (kNN)

ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli
"""

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):

    rotulos_teste = []

    for dado in dadosTeste:
        distancias = []
        for i in range(len(dadosTrain)):
            distancias.append(distancia_euclidiana(dado, dadosTrain[i]))

        distancias = list(enumerate(distancias))
        distancias = sorted(distancias, key=lambda x: x[1])

        rotulos = []
        for j in range(k):
            rotulos.append(rotuloTrain[distancias[j][0]])

        rotulos_teste.append(stats.mode(rotulos).mode[0])

    return rotulos_teste

def distancia_euclidiana(dadoTeste, dadoTrain):
    soma = 0
    for i in range(len(dadoTeste)):
        soma += (dadoTeste[i] - dadoTrain[i]) ** 2

    return math.sqrt(soma)

def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []
    for idx in range(0, len(dados)):
        if (rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])

    return ret

def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red', marker='^')
    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue', marker='+')
    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')

    plt.show()

def funcao_acuracia(rotulo_previsto, rotulo_esperado):
    total_corretos = 0
    for i in range(len(rotulo_previsto)):
        total_corretos += 1 if rotulo_previsto[i] == rotulo_esperado[i] else 0

    total_num = len(rotulo_esperado)

    return total_corretos / total_num

def main():
    mat = scipy.loadmat('grupoDados1.mat')

    grupoTest = mat['grupoTest']
    grupoTrain = mat['grupoTrain']
    testRots = mat['testRots']
    trainRots = mat['trainRots']

    rotulo_previsto = meuKnn(grupoTrain, trainRots, grupoTest, 10)
    print (funcao_acuracia(rotulo_previsto, testRots))

if __name__ == "__main__":
    main()