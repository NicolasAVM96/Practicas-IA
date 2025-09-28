import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
# 1) Se obtienen columnas numéricas (ej. fixed acidity, volatile acidity, citric acid, ... , calidad)
df = pd.read_csv(r"D:\Programacion\Python\IA\Practicas-IA\Eva2\control_calidad.csv")

# 2) Forzar a que todas las columnas (excepto las categóricas/texto) sean numéricas
df = df.apply(pd.to_numeric, errors="ignore")

# 3) Etiqueta PASA si calidad >= 6
df['PASA'] = (df['calidad'] >= 6).astype(int)
X = df.drop(columns=['calidad','PASA'])
y = df['PASA']

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5) Modelo
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# 6) Evaluación
y_pred = clf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# ------------------------ MATRIZ DE CONFUSIÓN (Dashboard) ------------------------
# 1) Calcular matriz
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
etiquetas = ["No Pasa", "Pasa"]  # 0=no pasa, 1=pasa

# 2) Métricas resumidas
tn, fp, fn, tp = cm.ravel()
total = cm.sum()
aciertos = tn + tp
errores = fp + fn
accuracy = aciertos / total if total else 0.0

# 3) Graficar matriz con números
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etiquetas)
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

# 4) Agregar porcentajes por celda (sobre el total)
for (i, j), val in np.ndenumerate(cm):
    pct = (val / total * 100) if total else 0.0
    ax.text(
        j, i + 0.25, f"{pct:.1f}%",  # un poquito abajo del número entero
        ha="center", va="center", fontsize=9, color="black"
    )

# 5) Títulos y ejes en español
ax.set_title("Matriz de Confusión - Control de Calidad", pad=15)
ax.set_xlabel("Predicción del Modelo")
ax.set_ylabel("Valor Real")

# 6) Recuadro con resumen (aciertos, errores, total, accuracy)
resumen = (
    f"Aciertos (TN+TP): {aciertos}\n"
    f"Errores  (FP+FN): {errores}\n"
    f"Total muestras : {total}\n"
    f"Accuracy       : {accuracy:.2%}"
) 
plt.gcf().text( #Ubicacion del resumen
    0.02, 0.95, resumen, fontsize=10, va="top", ha="left",
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#999")
)

plt.tight_layout()
plt.show()
# ------------------------ FIN DASHBOARD ------------------------

# 7) Importancia de variables
importancias = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
importancias.plot(kind='bar')
plt.title("Importancia de variables")
plt.ylabel("Importancia")
plt.show()

# 8) Mejora rápida: más profundidad ***POR IMPLEMENTAR****
# clf2 = DecisionTreeClassifier(max_depth=6, random_state=42)
# clf2.fit(X_train, y_train)
# y_pred2 = clf2.predict(X_test)
# print("Accuracy (modelo 2):", round(accuracy_score(y_test, y_pred2), 3))
