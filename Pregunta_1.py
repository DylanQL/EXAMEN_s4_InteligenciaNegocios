# pregunta_1.py

# Paso 1: Instalación de las librerías necesarias
# pip install ucimlrepo scikit-learn

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Paso 2: Cargar los datos
wine = fetch_ucirepo(id=109)

# Variables predictoras (X)
X = wine.data.features[['Alcohol', 'Alcalinity_of_ash', 'Nonflavanoid_phenols']]

# Variable target (y)
y = wine.data.targets

# Paso 3: Particionar los datos en un 80% para entrenamiento y un 20% para prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Crear y entrenar el modelo SVM
model = SVC(kernel='linear')  # Usamos el kernel lineal
model.fit(X_train, y_train)

# Paso 5: Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Paso 6: Evaluar el modelo usando dos métricas
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Paso 7: Mostrar los resultados
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")
