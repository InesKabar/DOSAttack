# Importation des packages

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


# Préparation du dataset

df = pd.read_csv(r'Normal.csv')       # Dataset normal
df1 = pd.read_csv(r'Dos2.csv')        # Dataset DoS

df['Label'] = 'Normal'                # Ajoute le label Normal
df1['Label'] = 'DoS'                 # Ajoute le label DoS

dataset = pd.concat([df, df1], ignore_index=True)

dataset.to_csv('dataset.csv', encoding='utf-8', index=False)


# Séparation entre les attributs et le label

X = dataset.drop('Label', axis=1)
Y = dataset['Label'].values


# Division du dataset en données d'apprentissage et de test

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=100,
    stratify=Y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# Apprentissage de l'algorithme avec l'arbre de décision

arbre = DecisionTreeClassifier(
    criterion='entropy',
    random_state=100,
    max_depth=3,
    min_samples_leaf=5
)

arbre.fit(X_train, Y_train)


# Prédictions sur les données de test

y_pred = arbre.predict(X_test)


# Matrice de confusion

cm = confusion_matrix(Y_test, y_pred, labels=['Normal', 'DoS'])

print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Normal', 'DoS']
)

cm_display.plot()
plt.show()


# Évaluation de l'arbre de décision

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Recall:", recall_score(Y_test, y_pred, pos_label='DoS'))
print("F1 score:", f1_score(Y_test, y_pred, pos_label='DoS'))


# Visualisation de l'arbre de décision

plt.figure(figsize=(15, 10))

plot_tree(
    arbre,
    feature_names=X_train.columns,
    class_names=['Normal', 'DoS'],
    filled=True
)

plt.title("Decision Tree - DoS Detection")
plt.show()
