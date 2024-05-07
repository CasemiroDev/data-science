import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Leitura da base de dados
data = os.getcwd()
database = os.path.join(data, './svc-1/database.csv')
df = pd.read_csv(database)

# Inputs e Outputs
X = df.drop('risco',axis=1)
y = df.risco
 
# Normalização
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)

# Divisão de treinos e testes
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, train_size=2/3, random_state=1)

# Aplicação do classificador
svc = SVC()
svc.fit(X_train, y_train)

# Accuracy
print(accuracy_score(y_test, svc.predict(X_test)))



