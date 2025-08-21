#Ejemplo de predicciones de casas con modelo de regresion lineal

x_train = [[1400, 3],[2000, 4],[1600, 3],[1900, 4]] #Tamaño en pies cuadrados y habitaciones
y_train = [200000, 300000, 250000, 280000] #Precio casas

#Se importa modelo de regresion lineal
from sklearn.linear_model import LinearRegression

#Crea una instancia del modelo de regresion lineal
model = LinearRegression()

#Ajusta el modelo a los datos de entrenamiento
model.fit(x_train, y_train)

#Se comienzan a hacer las predicciones
x_test = [[1800, 3], [2200, 4]] #Datos de prueba
predictions = model.predict(x_test)
print(predictions)