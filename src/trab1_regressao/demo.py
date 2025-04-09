import math
import matplotlib.pyplot as plt

"""
ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli

3)      Qual dos datasets não é apropriado para regressão linear?
R:  O segundo conjunto de dados não funciona bem com regressão linear porque os pontos seguem uma curva em vez de uma linha reta. 
    Se tentarmos ajustar uma linha reta a esses dados, ela não representará bem a relação entre 
    x e y, e as previsões ficarão erradas. 
    Nesse caso, uma equação que leva em conta essa curva, como uma regressão quadrática, funcionaria muito melhor.

"""
x1 = [10,8,13,9,11,14,6,4,12,7,5]
y1 = [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]

x2 = [10,8,13,9,11,14,6,4,12,7,5]
y2 = [9.14,8.14,8.47,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74]

x3 = [8,8,8,8,8,8,8,8,8,8,19]
y3 = [6.58,5.76,7.71,8.84,8.47,7.04,5.25,5.56,7.91,6.89,12.50]

def correlacao(x, y):

    media_x = sum(x) / len(x) 
    media_y = sum(y) / len(y)

    somatorio = 0

    somatorio_x = 0
    somatorio_y = 0

    for i in range(len(x)):
        somatorio += (x[i] - media_x) * (y[i] - media_y)

        somatorio_x += (x[i] - media_x)**2
        somatorio_y += (y[i] - media_y)**2 

    return somatorio/math.sqrt(somatorio_x * somatorio_y)

def regressao(x, y):

    media_x = sum(x) / len(x) 
    media_y = sum(y) / len(y)

    somatorio_cima = 0
    somatorio_baixo = 0

    for i in range(len(x)):
        somatorio_cima += (x[i] - media_x) * (y[i] - media_y)
        somatorio_baixo += (x[i] - media_x)**2

    beta_1 = somatorio_cima / somatorio_baixo
    beta_0 = media_y - (beta_1 * media_x)

    return (beta_0, beta_1)

def gera_graficos(x, y):

    correlacao_linear = correlacao(x, y)
    beta_0, beta_1 = regressao(x, y)

    menor_x = int(min(x)) - 1
    maior_x = int(max(x)) + 2
    array_x = []
    array_y = []

    for i in range(menor_x, maior_x):
        array_x.append(i)
        array_y.append(beta_0 + (beta_1 * i))

    plt.scatter(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Gráfico de Dispersão\n Regressão: y = {beta_0:.4f} + {beta_1:.4f}*x | Correlação: r = {correlacao_linear:.4f}")
    plt.plot(array_x, array_y, color="red")
    plt.show()

if __name__ == "__main__":
    gera_graficos(x1,y1)
    gera_graficos(x2,y2)
    gera_graficos(x3,y3)
