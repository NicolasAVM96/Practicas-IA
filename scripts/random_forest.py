from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#--------------------               Paso 1              --------------------#
# Carga el dataset de iris
iris = load_iris()
x = iris.data  # Características (variables independientes)
y = iris.target  # Etiquetas (variables dependientes)

#--------------------               Paso 2              --------------------#
# Divide el conjunto de datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#--------------------               Paso 3              --------------------#
# Crea y entrena el modelo de Random Forest
decision_tree = RandomForestClassifier(random_state=42)
decision_tree.fit(x_train, y_train)
# El modelo recibe las características de entrenamiento (x_train) → medidas de sépalos y pétalos.

#Rpedecit las etiquetas de clas para los datos de prueba usando el arbol de decision
y_pred_tree = decision_tree.predict(x_test)

#Calcular la precicisonn del modelo de arbol de decision
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print(f"Precisión del Árbol de decisión: {accuracy_tree * 100:.2f}%")

#--------------------               Paso 4              --------------------#
#Inicializar y entrenar el bosque aleatorio
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train, y_train)

#Predecir las etiquetas de clase para los datos de prueba usando el bosque aleatorio
y_pred_forest = random_forest.predict(x_test)

#Calcular la precisión del modelo de bosque aleatorio
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f"Precisión del Bosque Aleatorio: {accuracy_forest * 100:.2f}%")

