import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import warnings
from scipy.io import arff  
import os
from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore")

# Carpeta donde se encuentra el archivo ARFF y el archivo CSV resultante
data_folder = 'dataset'
arff_file_name = 'rice.arff'
arff_file = os.path.abspath(os.path.join(data_folder, arff_file_name))
csv_file_name = 'rice.csv'
csv_file = os.path.abspath(os.path.join(data_folder, csv_file_name))

# Cargar archivo ARFF y convertirlo a un DataFrame de Pandas
data, meta = arff.loadarff(arff_file)
df = pd.DataFrame(data)

# Asigna valores numéricos a las clases 'Cammeo' y 'Osmancik'
df['Class'] = df.Class.replace(b'Cammeo', 1)   # 1 para la clase 'Cammeo'
df['Class'] = df.Class.replace(b'Osmancik', 2) # 2 para la clase 'Osmancik'


df = shuffle(df)

df.to_csv(csv_file, index=False)

# Divide los datos en características (X) y etiquetas (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model1 = LogisticRegression(solver='sag', penalty='l2', max_iter=100)
model1.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba y entrenamiento
y_pred = model1.predict(X_test)
y_pred_train = model1.predict(X_train)

target = ['Cammeo', 'Osmancik']
print(classification_report(y_test, y_pred, target_names=target))

print(f'Accuracy of Model 1 on test data: {model1.score(X_test, y_test)}')

#muestra el error cuadrático medio en el conjunto de prueba y entrenamiento
print(f'Mean Squared Error of Model 1 on test data: {mean_squared_error(y_pred, y_test)}')
print(f'Mean Squared Error of Model 1 on training data: {mean_squared_error(y_pred_train, y_train)}')

#Graficas
train_sizes, train_scores, test_scores = learning_curve(model1, X, y)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Genera una matrix
confusion_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_mat, annot=True, cmap='Blues')
plt.title('Confusion Matrix (Model 1)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Gráfica de la curva de aprendizaje
plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score (Model 1)')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Score (Model 1)')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.xlabel('Number of Training Samples')
plt.ylabel('Score')
plt.title('Learning Curve (Model 1)')
plt.legend(loc='best')
plt.grid()

# Segundo modelo de regresión logística
model2 = LogisticRegression(solver='lbfgs', penalty='none', max_iter=5000)
model2 = model2.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba y entrenamiento para el segundo modelo
y_pred2 = model2.predict(X_test)
y_pred2_train = model2.predict(X_train)

print(classification_report(y_test, y_pred2, target_names=target))

print(f'Accuracy of Model 2 on test data: {model2.score(X_test, y_test)}')

print(f'Mean Squared Error of Model 2 on test data: {mean_squared_error(y_pred2, y_test)}')
print(f'Mean Squared Error of Model 2 on training data: {mean_squared_error(y_pred2_train, y_train)}')

# Gráfica de la matrix
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1)
confusion_mat2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(confusion_mat2, annot=True, cmap='Blues')
plt.title('Confusion Matrix (Model 2)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.subplot(1, 2, 2)
train_sizes, train_scores, test_scores = learning_curve(model2, X, y)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score (Model 2)')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-Validation Score (Model 2)')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='g')
plt.xlabel('Number of Training Samples')
plt.ylabel('Score')
plt.title('Learning Curve (Model 2)')
plt.legend(loc='best')
plt.grid()

plt.tight_layout()
plt.show()
