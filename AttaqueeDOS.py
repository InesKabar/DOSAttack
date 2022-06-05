
# Importation des packages 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier , plot_tree 
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt 



#Préparation du dataset 

df = pd.read_csv (r'Normal.csv')        #dataset du saus Attaque
df1 = pd.read_csv (r'Dos2.csv')       #dataset du Attaque DoS

df1['Label'] = 'DoS'          #ajoute colonne label avec valeur DoS
          
dataset = pd.concat([df,df1])                                   #fusionner les dataset DoS.csv et Normal.csv
dataset.to_csv('dataset.csv',encoding='utf-8', index= False)    #sauvgarder la dataset sous le nom dataset.csv



X = dataset[df1.columns]       #séparation entre tous les attributs et l'attribut Label
Y = dataset.Label.values       #divser dataset en train et test


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)   #70% apprentissage 30% test    
X_train.shape                                                         #nbr des base apprentissage  depuis dataset
X_test.shape                                                          #nbr des base test

#aprentisage de l'algorithme l'arbre de décision

arbre = DecisionTreeClassifier(criterion='entropy',random_state=100,max_depth=3,min_samples_leaf=5)
arbre.get_params()                                                                                      #description l'algorithme
arbre.fit(X_train,Y_train)                                                                              #Train decision tree classifer
y_pred = arbre.predict(X_test)                                                                          #faire des prédictions sur les données de test
print("nombre de prédictions corrects et incorrect ", confusion_matrix(Y_test,y_pred, labels=[0,1]))  
cm = confusion_matrix(Y_test,y_pred, labels=[0,1])
cm_display = ConfusionMatrixDisplay(cm).plot()

#évaluation l'arbre de décision 

print("Accuracy:",accuracy_score(Y_test,y_pred))
print("Recall:",recall_score(Y_test,y_pred))
print("F1 score:",f1_score(Y_test,y_pred))



#Visualisation de l'arbre de décision 

plt.figure()
plot_tree(arbre,feature_names=X_train.columns,filled=True)
plt.title("Decision tree")
plt.show()