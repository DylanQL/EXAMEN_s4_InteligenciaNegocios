import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos
data = pd.read_csv('aids_clinical.csv')

# Variable a predecir
y = data['str2']

# Variables predictoras (todas excepto str2)
X = data.drop(['str2'], axis=1)

# Eliminar la primera columna sin nombre si existe
if X.columns[0] == 'Unnamed: 0' or X.columns[0] == '':
    X = X.iloc[:, 1:]

# 1. Particionar la base en 75% Train y 25% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Utilizar el algoritmo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predicciones en data test
y_pred = rf_model.predict(X_test)

# 3. Dos métricas para evaluación del modelo
print("=" * 60)
print("EVALUACIÓN DEL MODELO RANDOM FOREST")
print("=" * 60)

# Métrica 1: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nMétrica 1 - Accuracy: {accuracy:.4f}")
print(f"Interpretación: El modelo clasificó correctamente el {accuracy*100:.2f}% de las muestras de prueba.")

# Métrica 2: Classification Report
print("\nMétrica 2 - Classification Report:")
print(classification_report(y_test, y_pred))
print("Interpretación:")
print("- Precision: Proporción de predicciones positivas que fueron correctas.")
print("- Recall: Proporción de casos positivos reales que fueron identificados.")
print("- F1-Score: Media armónica entre precision y recall.")
print("- Support: Número de muestras de cada clase en el conjunto de prueba.")
