import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Cargar datos
data = pd.read_csv('breast_wisconsin.csv')

# Variable a predecir
y = data['fractal_dimension3']

# Variables predictoras (todas excepto fractal_dimension3 y Diagnosis)
X = data.drop(['fractal_dimension3', 'Diagnosis'], axis=1)

# Eliminar la primera columna sin nombre si existe
if X.columns[0] == 'Unnamed: 0' or X.columns[0] == '':
    X = X.iloc[:, 1:]

# 1. Particionar la base en 80% Train y 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Utilizar el algoritmo de XGBoost
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predicciones en data test
y_pred = xgb_model.predict(X_test)

# 3. Dos métricas para evaluación del modelo
print("=" * 60)
print("EVALUACIÓN DEL MODELO XGBOOST")
print("=" * 60)

# Métrica 1: RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nMétrica 1 - RMSE: {rmse:.6f}")
print(f"Interpretación: El modelo tiene un error promedio de {rmse:.6f} unidades")
print(f"en la predicción de fractal_dimension3. Un RMSE menor indica mejor ajuste.")

# Métrica 2: R² (Coeficiente de Determinación)
r2 = r2_score(y_test, y_pred)
print(f"\nMétrica 2 - R²: {r2:.4f}")
print(f"Interpretación: El modelo explica el {r2*100:.2f}% de la variabilidad")
print(f"en fractal_dimension3. Un R² cercano a 1 indica excelente ajuste.")
