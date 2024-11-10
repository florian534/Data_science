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
  Le but est de connaître les sentiments des personnes par leurs tweets envers les compagnies aériennes américaines. L'ensemble de données comporte 14640 tweets, les données de Twitter ont été extraites de février 2015 et les contributeurs ont été invités à classer les tweets positifs, négatifs et neutres, 

#### Techniques utilisées :
- Traitement du langage naturel (NLP)
- Modèles de classification supervisée : RandomForestClassifier

#### Résultats :
- **accuracy score** : 0.671448087431694
- **Matrice de confusion** 

![image](https://github.com/user-attachments/assets/2432aac6-917b-485b-bc67-52030ba4904e)

### c. Régression : **Technique de régression avancée des prix de l'immobilier**
Le projet Kaggle "House Prices - Advanced Regression Techniques" est un concours populaire qui consiste à prédire les prix de vente de maisons dans la ville d'Ames, Iowa, en utilisant diverses techniques de régression. Ce concours offre un excellent point d'entrée pour les praticiens de la science des données afin d'explorer des modèles de régression avancés, de manipuler des données et de comprendre la relation entre différentes caractéristiques des maisons et leurs prix de vente. On a un ensemble de données de plus de 1460 lignes, 80 variables + la variable cible.

#### Techniques utilisées :
- Traitement des valeurs aberrantes (méthode : Z_scores et QQ plot)
- Normalisez les données
- Sélection des variables (matrice de corrélation, VIF)
- Encodage des données qualitatifs
- Modèles de classification supervisée : XGBoost

#### Résultats :
- **RMSE score** : 0.24954
- **soumission des prédictions sur kaggle** 
![image](https://github.com/user-attachments/assets/e7e58776-98e1-4eaf-906d-0a0d3d7c07b3)


## 2. Machine Learning : Apprentissage non supervisé
### a. Analyse de données sur la Production d'électricité**

Le but de notre étude était donc de faire une segmentation expliquée des jours en fonction de la plage de puissance d'injection (kw).


#### Techniques utilisées :
- Data engineering (groupby,normalisation,boxplot)
- La classification hiérarchique (CAH)
- Analyse en composante principale (ACP)
  
#### Résultats :
Représentation graphique de l'ACP

![image](https://github.com/user-attachments/assets/9eaeeca5-63f8-4be7-bb1c-b31fba8bc63c)


Pour notre analyse,il a été pertinent d'effectuer l'analyse par mois afin d'examiner si, au cours des différentes saisons (hiver, printemps, été, automne), la plage de puissance injectée présentent des regroupements d'observations cohérents. En analysant la répartition de la puissance injectée selon les mois, nous constatons une concentration des observations sur la partie gauche du graphique pour les mois d'octobre, novembre, décembre et janvier. Cette tendance suggère une augmentation de la puissance injectée durant cette période hivernale. En revanche, pour les mois d'avril, mai et juin, les points sont majoritairement regroupés sur la partie droite du graphique, ce qui pourrait indiquer une diminution de la puissance injectée au printemps.

## 3. Deep learning
### a. Réseau de neurones convolutionnels : chihuahua or muffin**
La base de données "Chihuahua or Muffin" est une collection amusante et populaire utilisée dans le contexte de la vision par ordinateur et des applications de deep learning. Cette base de données est un exemple classique de défi de classification d'images où l'objectif est de différencier des photos de chiens Chihuahua de celles de muffins, en raison de leur ressemblance surprenante.

#### Techniques utilisées et résultats d'accuracy sur jeux de données test :
- Convolution neural network (CNN) + dropout : 0.9507042169570923
- Transfert learning (Model VGG16) : 0.9929577464788732
- Fine Tuning (ResNet50 / geler les couches) : 0.9929577708244324
- Keras Tuner : 0.625

### b. Projet fil rouge : détection d'aliments sur les images
Ce projet est un fil rouge réalisé dans le cadre du mastère spécialisé "Expert en Science des Données" à l'INSA de Rouen. L'idée est de se placer dans le contexte où une personne dispose d'un ensemble d'ingrédients et souhaite créer une nouvelle recette qui n'existe pas encore. L'objectif est de permettre l'invention d'un plat inédit, potentiellement délicieux, en utilisant les ingrédients disponibles.

Pour ce faire, nous avons développé un modèle qui, à partir d'une image d'ingrédients, détecte et identifie les éléments présents. Ces ingrédients seront ensuite utilisés pour générer une nouvelle recette (algorithme gpt-2). Dans ce projet, j'ai pris en charge la création de la base de données ainsi que la détection des aliments sur les images. 
#### Techniques utilisées:
- Etat de l'art sur la reconnaissance des ingrédients
- Création de la base de données (web scraping sur google images + données kaggle + fusion des images)
- Modèle de détection Yolov3
- Modèle de détection ssdlite_mobilnet_v2
#### Résultats :
Détection sur une image avec ssdlite_mobilnet_v2

![image](https://github.com/user-attachments/assets/318b17d3-4a7a-439b-a5be-d4a2692a2d41)

- L’image nous informe qu’il y a:
  - 89 % de chances que l’ingrédient détecté soit une orange
  - 97 % de chance que l’ingrédient détecté soit une banane
  - 40 % de chance que l’ingrédient détecté soit une pomme.

## 1. Large Language Model (LLM)

### b. Avis des tweets (NLP)
Le but est de connaître les sentiments des personnes par leurs tweets envers les compagnies aériennes américaines. L'ensemble de données comporte 14640 tweets, les données de Twitter ont été extraites de février 2015, on se met dans le cas où on n'a pas de label, on utilise un algorithme de LLM avec huggline face qui nous donne un score de 1 à 5 sur l'avis du tweet.



- Utilisation du LLM (BERT) pour la classification de sentiments des données textuelles (NLP).
On se met dans le cas où on n'a pas de label, on utilise un algorithme de LLM avec huggline face qui nous donne un score de 1 à 5 sur l'avis du tweet.
- Utilisation du RAG avec le modèle LLM (LLaMa2).

Création API avec Flask :
- Jeux de données Iris.


Des projets en optimisation (python/cplex) : 
- Construction d'un stade dans les meilleurs délais.
- Theorie des graphes : algorithme de Dijkstra, plus proche voisin.
- Optimistion avancée : Heuristique pour le problème du voyageur de commerce , optimisation stochastique (Engie : mix électrique renouvelable en 2050).
