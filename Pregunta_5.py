from pyspark.sql import SparkSession
from pyspark.sql.functions import sum

# Crear sesión de Spark
spark = SparkSession.builder.appName("Pregunta5").getOrCreate()

# Cargar datos
df = spark.read.csv('data.csv', header=True, inferSchema=True)

print("=" * 60)
print("PREGUNTA 5 - PYSPARK")
print("=" * 60)

# 1. Mostrar el tipo de datos de todas las variables
print("\n1. Tipo de datos de todas las variables:")
df.printSchema()

# 2. Mostrar sólo las columnas "CIC" y "EGFR"
print("\n2. Columnas CIC y EGFR:")
df.select("CIC", "EGFR").show()

# 3. Mostrar la suma total de la variable "EGFR"
suma_egfr = df.select(sum("EGFR")).collect()[0][0]
print(f"\n3. Suma total de EGFR: {suma_egfr}")

# 4. Mostrar la suma total de la variable "SMARCA4"
suma_smarca4 = df.select(sum("SMARCA4")).collect()[0][0]
print(f"\n4. Suma total de SMARCA4: {suma_smarca4}")

# Detener sesión de Spark
spark.stop()
