#importando as bibliotecas necessárias
import pandas as pd
import numpy as np

#Os dados da questão foram arquivados em um arquivo 
base = pd.read_csv("base-questao-2-final-exel.csv")

#Criando a variável X e atribuindo as colunas 0 e 1 a ela e a coluna 2 a y
X = base.iloc[:, 0:2].values
y = base.iloc[:, 2].values

#Fazendo a divisão entre bases de treinamento e bases de teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Criando um modelo de regressão
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#R² para o treinamento
score = regressor.score(X_train, y_train)

previsoes = regressor.predict(X_test)

#Erro absoluto
from sklearn.metrics import mean_absolute_error
#y_teste é a resposta real, previsões é o teste
mae = mean_absolute_error(y_test, previsoes)

#R² na base de teste
score2 = regressor.score(X_test, y_test)

#Visualizando os parametros
regressor.intercept_
regressor.coef_






