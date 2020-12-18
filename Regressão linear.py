#importando as bibliotecas utilizadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#atribuindo os dados do gerador a base
base = pd.read_csv("aerogerador.csv")

#atribuindo a coluna de indice 0 a x e de indice 1 a y
X = base.iloc[:, 0].values
y = base.iloc[:, 1].values
#analizando a relação entre as colunas
correlacao = np.corrcoef(X, y)

X = X.reshape(-1, 1)

#Regressão Polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 5)
X_poly = poly.fit_transform(X)

regressor = LinearRegression()
regressor.fit(X_poly, y)

#R² e R²aj
score = regressor.score(X_poly, y)
score_ad = 1 - (1-score)*(len(X)-1)/(len(X)-X.shape[1]-1)

#plotando o gráfico
plt.scatter(X, y)
plt.plot(X, regressor.predict(poly.fit_transform(X)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Velocidade do vento')
plt.ylabel('Potência gerada')
