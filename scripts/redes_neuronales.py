import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical


#--------------------               Paso 1              --------------------#
#Carga datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#--------------------               Paso 2              --------------------#
#Se procesan los datos
x_train = x_train / 255.0 
x_test =  x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#--------------------               Paso 3              --------------------#
#Se crea el modelo
model = Sequential()
model.add(Flatten(input_shape=(28, 28))) #Convierte la imagen 2D en un vector 1D
model.add(Dense(128, activation='relu')) #Capa oculta con 128 neuronas y funcion de activacion ReLU
model.add(Dense(10, activation='softmax')) #Capa de salida con 10

#--------------------               Paso 4              --------------------#
#Compilacion del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Entrenamiento del modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

#Evaluacion del modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%")