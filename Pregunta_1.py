from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Fetch dataset
wine = fetch_ucirepo(id=109)

# Data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

# Seleccionar variables predictoras
X_selected = X[['Alcohol', 'Alcalinity_of_ash', 'Nonflavanoid_phenols']]

# 1. Particionar la base en 80% Train y 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 2. Usar el algoritmo de SVM
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train.values.ravel())

# Predicciones en data test
y_pred = svm_model.predict(X_test)

# 3. Dos métricas para evaluación del modelo
print("=" * 60)
print("EVALUACIÓN DEL MODELO SVM")
print("=" * 60)

# Métrica 1: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nMétrica 1 - Accuracy: {accuracy:.4f}")
print(f"Interpretación: El modelo clasificó correctamente el {accuracy*100:.2f}% de las muestras de prueba.")

# Métrica 2: Classification Report (Precision, Recall, F1-Score)
print("\nMétrica 2 - Classification Report:")
print(classification_report(y_test, y_pred))
print("Interpretación:")
print("- Precision: Proporción de predicciones positivas que fueron correctas.")
print("- Recall: Proporción de casos positivos reales que fueron identificados.")
print("- F1-Score: Media armónica entre precision y recall.")
print("- Support: Número de muestras de cada clase en el conjunto de prueba.")
