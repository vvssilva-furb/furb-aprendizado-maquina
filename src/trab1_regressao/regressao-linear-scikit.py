import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dados extraídos da imagem
dados = np.array([
    [2104, 3, 399900], [1600, 3, 329900], [2400, 3, 369000], [1416, 2, 232000],
    [3000, 4, 539900], [1985, 4, 299900], [1534, 3, 314900], [1427, 3, 199000],
    [1380, 3, 212000], [1494, 3, 242500], [1940, 4, 240000], [2000, 3, 347000],
    [1890, 3, 330000], [4478, 5, 699900], [1268, 3, 259900], [2300, 4, 449900],
    [1320, 2, 299900], [1236, 3, 199900], [2609, 4, 500000], [3031, 4, 599000],
    [1767, 3, 252900], [1888, 2, 255000], [1604, 3, 242900], [1962, 4, 259900],
    [3890, 3, 573900], [1100, 3, 249900], [1458, 3, 464500], [2526, 3, 469000],
    [2200, 3, 475000], [2637, 3, 299900], [1839, 2, 349900], [1000, 1, 169900],
    [2040, 4, 314900], [3137, 3, 579900], [1811, 4, 285900], [1437, 3, 249900],
    [1239, 3, 229900], [2132, 4, 345000], [4215, 4, 549000], [2162, 4, 287000],
    [1664, 2, 368500], [2238, 3, 329900], [2567, 4, 314000], [1200, 3, 299000],
    [852, 2, 179900], [1852, 4, 299900], [1203, 3, 239500]
])

# Criando DataFrame
df = pd.DataFrame(dados, columns=["Tamanho", "Quartos", "Preco"])

# Cálculo da correlação
correlacao_tamanho_preco = df["Tamanho"].corr(df["Preco"])
correlacao_quartos_preco = df["Quartos"].corr(df["Preco"])

# Regressão linear com Tamanho e Preço
modelo_tamanho = LinearRegression()
modelo_tamanho.fit(df[["Tamanho"]], df["Preco"])

# Regressão linear com Quartos e Preço
modelo_quartos = LinearRegression()
modelo_quartos.fit(df[["Quartos"]], df["Preco"])

# Regressão linear múltipla
X = df[["Tamanho", "Quartos"]]
y = df["Preco"]
# alterar train_Test
modelo = LinearRegression()
modelo.fit(X, y)

# Coeficientes (Beta)
beta_tamanho, beta_quartos = modelo.coef_
intercepto = modelo.intercept_

# Previsão para uma casa com tamanho 1650 e 3 quartos
previsao = modelo.predict([[1650, 3]])[0]

# Resultados
print("Correlação entre Tamanho e Preço:", correlacao_tamanho_preco)
print("Correlação entre Quartos e Preço:", correlacao_quartos_preco)
print("Beta Tamanho:", beta_tamanho)
print("Beta Quartos:", beta_quartos)
print("Intercepto:", intercepto)
print("Preço previsto para uma casa com 1650m² e 3 quartos:", previsao)
