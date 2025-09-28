#Ejemplo de Maquina de Soporte Vectorial (SVM)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#--------------------               Paso 1              --------------------#
 #Se cargan los datos Iris (Iris es un dataset que tiene 150 muestras (flores), cada una con 4 características.
iris = datasets.load_iris()
x = iris.data # Se guardan las caracteristicas de las flores en la var "X" / Son variables independientes
y = iris.target # Se cargan las especies de las flores / Son variables dependientes



#--------------------               Paso 2              --------------------#
# Se divide el conjunto de datos en entrenamiento y prueba
# Para entrenar el modelo, no se usa la totalidad de los datos, por lo cual se dejan las variables x_train y y_train como datos de entrenamiento
# Mientras que las variables x_test e y_test, se dejan para hacer pruebas, de esta manera podemos ver si el modelo realmente aprendio, ya que 
# Nunca vio los datos guardados en las variables de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
# Se le entregan como parametros a la funcion "train_test_split" las variables "x" e "y"
# El "test_size" es la cantidad en % de los datos con los cuales va a testear el modelo, en este caso es un 30% (0.3)
# El "random_state" se usa para q los resultados sean lo mismos en casos de practicas, asi todos los que prueban esto, obtienen el mismo resultado



#--------------------               Paso 3              --------------------#
# Se escalan datos
# "StandardScaler" escala los datos para que las variables mas grandes no dominen el calculo
sc = StandardScaler()

# "Fit" calcula la media y la desviacion estandar, mientras q "transform" usa los valores obtenidos por "fit" para escalar dichos valores, 
# dando numeros normalizados. En resumen aprende a como escalar y aplica el escalado
x_train = sc.fit_transform(x_train) 
# Porque se usa el "fit_transform"? 
# Se usa ya que el "x_train" son los datos para entrenar el modelo, a diferencia el "x_test", que son los datos de pruba

 #En este caso no se calcula la desviacion estandar ni la media, solo usa los valores aprendidos con "x_train" para aplicar el escalado en el "x_test"
x_test = sc.transform(x_test)



#--------------------               Paso 4              --------------------#
# Inicia y entrena el modelo SVM
svm_classifier = SVC(kernel="linear", random_state=42) # Se crea el modelo vacio "svm_classifier" con un "SVC", el cual clasificara y separara x clases de datos. 
# Al "SVC" se le entregan los parametros correspondientes, en este caso se usa el "linear"(lineal), ya que la separacion de las clases es bastante notoria, por lo cual
# La linea (o vector) de separacion es recta y no hay necesidad de usar una linea curva
# A este punto solo se crea el modelo y se le entregan los parametros, pero aun no se le dan los datos para trabajar

# Se entrena el modelo con el metodo ".fit" para ajustar el modelo de los datos
svm_classifier.fit(x_train, y_train)
# El modelo recibe las características de entrenamiento (x_train) → medidas de sépalos y pétalos.
# Recibe también las etiquetas de esas flores (y_train) → especie de cada flor.

#El SVM calcula: La línea/plano que mejor separa las distintas especies de flores y trata de dejar un margen amplio entre clases para que nuevas flores se clasifiquen bien.



#--------------------               Paso 5              --------------------#
# Predice las etiquetas de clase para datos de prueba
y_pred = svm_classifier.predict(x_test)



#--------------------               Paso 6              --------------------#
# Calcula la precision del modelo comparando lo que el modelo predijo ("y_pred") contra las etiquetas reales ("y_test")
precision = accuracy_score(y_test, y_pred)
print(f"La precisión del modelo SVM es: {precision*100:.2f}%") #Se multiplica la precision por 100 para dejarlo como %

