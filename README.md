# Mes Projets Académiques et Personnels

Dans ce répertoire se trouvent mes projets académiques et personnels, principalement codés en Python et R. Voici quelques exemples de projets que j'ai réalisés :

## 1. Machine Learning : Apprentissage Supervisé

### a. Classification : **Fraude sur les Cartes de Crédit**
Ce projet utilise un ensemble de données contenant des transactions effectuées par carte de crédit en septembre 2013 par des titulaires de carte européens. L'ensemble de données comporte 284 807 transactions, dont 492 sont frauduleuses. Le déséquilibre est important, la classe positive (fraude) représentant seulement 0,172% des transactions.

#### Objectif :
Détecter les fraudes sur les cartes de crédit en utilisant des techniques de classification supervisée.

#### Modèle utilisé :
- **Régression Logistique (Logistic Regression)**
  
#### Résultats :
- **AUC (Area Under the Curve)** : 0.9782
- **Courbe ROC** : Affichage du taux de vrais positifs en fonction du taux de faux positifs
  
  ![Courbe Roc](https://github.com/user-attachments/assets/06bea6fc-4cc3-4d83-9947-58148e06df5f)
  
Le meilleur modèle est la régression logistique (lr) avec un AUC (Area Under the Curve)  de 0.9782246384308207

### b. Classification : **Classification de Sentiments (NLP)**
Dans ce projet, l'objectif était d'analyser des textes pour classifier les sentiments exprimés (positifs, négatifs, neutres).

#### Techniques utilisées :
- Traitement du langage naturel (NLP)
- Modèles de classification supervisée : RandomForestClassifier

#### Résultats :
- **accuracy score** : 0.671448087431694
- **Matrice de confusion** 

![image](https://github.com/user-attachments/assets/2432aac6-917b-485b-bc67-52030ba4904e)

- Regression : Technique de régression avancée des prix de l'immobilier

En machine learning : apprentissage non supervisée 
- Analyse de données sur la Production d'électricité

En deep learning : 
- Réseau de neurones convolutionnels : chihuahua or muffin (model VGG16,Fine tuning, keras tuner).
- Détection d'aliments sur les images (Yolo v3) et génération de nouvelles recettes (GPT2): projet file rouge INSA.
- Utilisation du LLM (BERT) pour la classification de sentiments des données textuelles (NLP).
- Utilisation du RAG avec le modèle LLM (LLaMa2).

Création API avec Flask :
- Jeux de données Iris.

Système de recommendation :
- Netflix, MoviLens.

Des projets en optimisation (python/cplex) : 
- Construction d'un stade dans les meilleurs délais.
- Theorie des graphes : algorithme de Dijkstra, plus proche voisin.
- Optimistion avancée : Heuristique pour le problème du voyageur de commerce , optimisation stochastique (Engie : mix électrique renouvelable en 2050).
