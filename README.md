# ---VERSION EN FRANÇAIS---
# Mes projets académiques et personnels

Dans ce répertoire se trouvent mes projets académiques et personnels, principalement codés en Python et R. Voici quelques exemples de projets que j'ai réalisés :

## 1. Machine Learning : Apprentissage Supervisé

### a. Classification : **Fraude sur les Cartes de Crédit**
Ce projet utilise un ensemble de données contenant des transactions effectuées par carte de crédit en septembre 2013 par des titulaires de carte européens. L'ensemble de données comporte 284 807 transactions, dont 492 sont frauduleuses. Le déséquilibre est important, la classe positive (fraude) représentant seulement 0,172% des transactions.

#### Objectif :
Détecter les fraudes sur les cartes de crédit en utilisant des techniques de classification supervisée.

#### Modèle utilisé :
- **Régression Logistique**
- **K-plus proches voisin**
- **Arbre de décision**
- **Forêts aléatoires**
  
  
#### Résultats :
- **AUC (Area Under the Curve)** : 0.9782
- **Courbe ROC** : Affichage du taux de vrais positifs en fonction du taux de faux positifs
  
  ![Courbe Roc](https://github.com/user-attachments/assets/06bea6fc-4cc3-4d83-9947-58148e06df5f)
  
Le meilleur modèle est la régression logistique (lr) avec un AUC (Area Under the Curve)  de 0.9782246384308207

### b. Classification : **Classification de Sentiments (NLP)**
  Le but est de connaître les sentiments des personnes par leurs tweets envers les compagnies aériennes américaines. L'ensemble de données comporte 14640 tweets, les données de Twitter ont été extraites de février 2015 et les contributeurs ont été invités à classer les tweets positifs, négatifs et neutres, 

#### Techniques utilisées :
- Traitement du langage naturel (NLP) : mettre en minuscule, enlever les caractères spéciaux et les ponctuations, tokenisation, retrait des stopwords, réduire les différences grammaticales (stemming), bag wof word
- Modèle de classification supervisé : RandomForestClassifier

#### Résultats :
- **accuracy score** : 0.671448087431694
- **Matrice de confusion** 

![image](https://github.com/user-attachments/assets/17d71b35-3979-45cc-a642-0201615f008c)


### c. Régression : **Technique de régression avancée des prix de l'immobilier**
Le projet Kaggle "House Prices - Advanced Regression Techniques" est un concours populaire qui consiste à prédire les prix de vente de maisons dans la ville d'Ames, Iowa, en utilisant diverses techniques de régression. Ce concours offre un excellent point d'entrée pour les praticiens de la science des données afin d'explorer des modèles de régression avancés, de manipuler des données et de comprendre la relation entre différentes caractéristiques des maisons et leurs prix de vente. On a un ensemble de données de plus de 1460 lignes, 80 variables + la variable cible.

#### Techniques utilisées :
- Traitement des valeurs aberrantes (méthode : Z_scores et QQ plot)
- Normalisez les données (centrée réduite)
- Sélection des variables (matrice de corrélation, VIF)
- Encodage des données qualitatifs
- Modèle de classification supervisé : XGBoost

#### Résultats :
- **RMSE score** : 0.24954
- **soumission des prédictions sur kaggle** 
![image](https://github.com/user-attachments/assets/e7e58776-98e1-4eaf-906d-0a0d3d7c07b3)

- **Analyse et interprétation du modèle XGBoost à l'aide de SHAP** 
![image](https://github.com/user-attachments/assets/f0d14d39-1f8d-4bcc-8234-b2a4fdb7222d)

OverallQual est une caractéristique cruciale dans ce modèle : une maison avec une qualité globale élevée (valeur élevée pour OverallQual, représentée en rouge) augmente le prix de vente prédictif, tandis qu'une maison de qualité plus faible (valeur faible, en bleu) diminue ce prix.



## 2. Machine Learning : Apprentissage non supervisé
### a. Analyse de données sur la Production d'électricité

On doit analyser les données sur la production d'électricité. Le but de notre étude est de faire une segmentation expliquée des jours en fonction de la plage de puissance d'injection (kw).


#### Techniques utilisées :
- Data engineering (groupby, normalisation, boxplot)
- La classification hiérarchique (CAH)
- Analyse en composante principale (ACP)
  
#### Résultats :
Représentation graphique de l'ACP

![image](https://github.com/user-attachments/assets/9eaeeca5-63f8-4be7-bb1c-b31fba8bc63c)


Pour notre analyse, il a été pertinent d'effectuer l'analyse par mois afin d'examiner si, au cours des différentes saisons (hiver, printemps, été, automne), la plage de puissance injectée présentent des regroupements d'observations cohérents. En analysant la répartition de la puissance injectée selon les mois, nous constatons une concentration des observations sur la partie gauche du graphique pour les mois d'octobre, novembre, décembre et janvier. Cette tendance suggère une augmentation de la puissance injectée durant cette période hivernale. En revanche, pour les mois d'avril, mai et juin, les points sont majoritairement regroupés sur la partie droite du graphique, ce qui pourrait indiquer une diminution de la puissance injectée au printemps.

## 3. Deep learning
### a. Réseau de neurones convolutionnels : chihuahua or muffin
La base de données "Chihuahua or Muffin" est une collection amusante et populaire utilisée dans le contexte de la vision par ordinateur et des applications de deep learning. Cette base de données est un exemple classique de défi de classification d'images où l'objectif est de différencier des photos de chiens Chihuahua de celles de muffins, en raison de leur ressemblance surprenante.

#### Techniques utilisées et résultats d'accuracy sur le jeu de données test :
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
### c. Reconnaissance des sons d'oiseaux
Le Bird Song Dataset sur Kaggle est une collection de sons d'oiseaux, souvent utilisée pour des projets de classification audio et d'apprentissage automatique. Ce jeu de données contient des enregistrements audio de chants d'oiseaux de différentes espèces, ce qui en fait un excellent outil pour les chercheurs et les développeurs qui souhaitent travailler sur des tâches d'analyse de données audio, telles que la reconnaissance des espèces d'oiseaux par leur chant.

#### Techniques utilisées
- calculez le spectrogramme via la STFT (Transformée de Fourier à court terme)
- convertir ce spectrogramme en décibels
- normaliser les valeurs et donner le spectrogramme normalisé et l'étiquette associée
- Réseau de neurones convolutionnels

#### Résultats :
Evaluation du modèle

![image](https://github.com/user-attachments/assets/8d4de32f-fed4-4051-9dd7-7b807e8adb5d)

Matrice de confusion

![image](https://github.com/user-attachments/assets/f2c02ac9-1137-4ee6-afd9-edf934dba314)

Score 

F1: 0.9510736108330955 | Precision: 0.9504173755284869 | Recall: 0.9524755155991809 | AUC: 0.9979307755194758

## 4. Large Language Model (LLM)

### a. Avis des tweets (NLP)
Le but est de connaître les sentiments des personnes par leurs tweets envers les compagnies aériennes américaines. L'ensemble de données comporte 14640 tweets, les données de Twitter ont été extraites de février 2015, on se met dans le cas où on n'a pas de label, on utilise un algorithme de LLM (BERT) avec hugging face qui nous donne un score de 1 à 5 sur l'avis du tweet.

#### Résultats :
![9migpjpi](https://github.com/user-attachments/assets/8b10c5c9-ebd1-4cca-aefb-91c7761515b8)


### b. Utilisation du RAG avec le modèle LLM (LLaMa2)

Le RAG (Retrieval-Augmented Generation) combine la recherche d'informations et la génération de texte pour améliorer la précision des réponses en s'appuyant sur des données externes. En intégrant des documents pertinents avant la phase de génération, il permet de réduire les hallucinations et d'offrir des réponses plus factuelles et actualisées. J'ai réalisé un test en utilisant comme fichier PDF ma thèse professionnelle sur l'étude et implémentation des méthodes d'interprétabilité dans les modèles de tarification automobile.

## 5. Création d'interface de programmation (API)

### a. Localisation et recommandation de restaurants
Dans ce projet, j'ai utilisé MongoDB et Python pour travailler sur une base de données populaire de restaurants de New York. Cette base de données contient des informations telles que le nom, l'adresse, le type de cuisine, la localisation géographique (coordonnées latitude et longitude), ainsi que des évaluations des restaurants, incluant des scores et des grades attribués par les clients. J'ai développé une carte interactive permettant de visualiser la localisation des restaurants, et intégré des filtres dynamiques pour que l'utilisateur puisse affiner sa recherche en fonction du type de cuisine, du score maximum et du grade des restaurants.

![image](https://github.com/user-attachments/assets/49c9096d-ed43-44cc-b6a1-ef771e3c3ec5)



![image](https://github.com/user-attachments/assets/aca78036-a8f7-486a-ac68-ac57786d6636)


## b.Jeux de données Iris 


Le jeu de données Iris est l'un des ensembles de données les plus célèbres en machine learning. Il contient 150 observations de fleurs appartenant à trois espèces différentes : Iris-setosa, Iris-versicolor et Iris-virginica. Chaque observation est décrite par 4 caractéristiques :


-Longueur des sépales (en cm)
-Largeur des sépales (en cm)
-Longueur des pétales (en cm)
-Largeur des pétales (en cm)


Le but est souvent de prédire l'espèce de la fleur en se basant sur ces caractéristiques. Ce jeu de données est couramment utilisé pour des tâches de classification et pour tester différents algorithmes d'apprentissage supervisé.

![image](https://github.com/user-attachments/assets/3d7d3aa0-2220-45f7-b075-87b7b25a97c5)



## 6. Des projets en optimisation 
- Construction d'un stade dans les meilleurs délais. (cplex/python)
- Theorie des graphes : algorithme de Dijkstra, plus proche voisin.
- Optimistion avancée : Heuristique pour le problème du voyageur de commerce , optimisation stochastique (Engie : mix électrique renouvelable en 2050).





# ---ENGLISH VERSION--

My Academic and Personal Projects
This repository contains my academic and personal projects, mostly coded in Python and R. Below are some examples of projects I have worked on:

## 1. Machine Learning: Supervised Learning
### a. Classification: Credit Card Fraud Detection
This project uses a dataset containing transactions made by European credit card holders in September 2013. The dataset consists of 284,807 transactions, of which 492 are fraudulent. The imbalance is significant, with the positive class (fraud) representing only 0.172% of the transactions.

### Goal:
Detect credit card fraud using supervised classification techniques.

### Models used:
Logistic Regression
K-Nearest Neighbors
Decision Tree
Random Forest
Results:
AUC (Area Under the Curve): 0.9782

ROC Curve: Displaying the True Positive Rate against the False Positive Rate

![Courbe Roc](https://github.com/user-attachments/assets/06bea6fc-4cc3-4d83-9947-58148e06df5f)

The best model is Logistic Regression (LR) with an AUC of 0.9782246384308207.

### b. Classification: Sentiment Classification (NLP)
The goal is to determine the sentiment of individuals based on their tweets towards U.S. airlines. The dataset contains 14,640 tweets, extracted from Twitter data in February 2015, and contributors were asked to label the tweets as positive, negative, or neutral.

Techniques used:
Natural Language Processing (NLP): Lowercasing, removing special characters and punctuation, tokenization, stopword removal, stemming, bag of words
Supervised classification model: RandomForestClassifier
Results:
Accuracy score: 0.671448087431694
Confusion Matrix

![image](https://github.com/user-attachments/assets/17d71b35-3979-45cc-a642-0201615f008c)

### c. Regression: Advanced House Price Regression Techniques
The Kaggle project "House Prices - Advanced Regression Techniques" is a popular competition where the goal is to predict house sale prices in Ames, Iowa, using various regression techniques. This competition is an excellent entry point for data science practitioners to explore advanced regression models, manipulate data, and understand the relationship between various house features and their sale prices. The dataset contains over 1460 rows, with 80 variables + the target variable.

Techniques used:
Outlier handling (methods: Z-scores and QQ plot)
Data normalization (standardization)
Feature selection (correlation matrix, VIF)
Encoding categorical data
Supervised classification model: XGBoost
Results:
RMSE score: 0.24954

Kaggle submission predictions
![image](https://github.com/user-attachments/assets/e7e58776-98e1-4eaf-906d-0a0d3d7c07b3)

Analysis and interpretation of the XGBoost model using SHAP
![image](https://github.com/user-attachments/assets/f0d14d39-1f8d-4bcc-8234-b2a4fdb7222d)

OverallQual is a crucial feature in this model: a house with a high overall quality (represented in red) increases the predicted sale price, while a lower quality house (represented in blue) decreases the predicted price.

## 2. Machine Learning: Unsupervised Learning
### a. Electricity Production Data Analysis
The task is to analyze electricity production data. The goal of the study is to segment the days based on the power injection range (kW).

### Techniques used:
Data engineering (groupby, normalization, boxplot)
Hierarchical clustering (HAC)
Principal Component Analysis (PCA)
Results:
PCA graphical representation
![image](https://github.com/user-attachments/assets/9eaeeca5-63f8-4be7-bb1c-b31fba8bc63c)


For our analysis, it was relevant to perform the analysis by month to examine if the power injection range shows consistent groupings of observations across different seasons (winter, spring, summer, fall). By analyzing the distribution of power injection by month, we notice a concentration of observations on the left side of the chart for the months of October, November, December, and January. This suggests an increase in power injection during the winter period. Conversely, for the months of April, May, and June, the points are mostly concentrated on the right side of the chart, indicating a decrease in power injection during the spring.

## 3. Deep Learning
### a. Convolutional Neural Network: Chihuahua or Muffin
The "Chihuahua or Muffin" dataset is a fun and popular collection used in computer vision and deep learning applications. This dataset is a classic image classification challenge where the goal is to distinguish photos of Chihuahua dogs from those of muffins, due to their surprising resemblance.

### Techniques used and accuracy results on the test dataset:
Convolutional Neural Network (CNN) + Dropout: 0.9507042169570923
Transfer Learning (Model VGG16): 0.9929577464788732
Fine-Tuning (ResNet50 / freezing layers): 0.9929577708244324
Keras Tuner: 0.625
### b. Capstone Project: Food Detection in Images
This project was a capstone project in the "Expert in Data Science" master's program at INSA Rouen. The idea is to place ourselves in a context where a person has a set of ingredients and wishes to create a new recipe that does not yet exist. The goal is to invent a potentially delicious dish using the available ingredients.

To do this, we developed a model that detects and identifies ingredients from an image. These ingredients will then be used to generate a new recipe (GPT-2 algorithm). In this project, I handled creating the database and detecting food items in the images.

### Techniques used:
State-of-the-art on ingredient recognition
Database creation (web scraping from Google Images + Kaggle data + image fusion)
Yolov3 detection model
ssdlite_mobilnet_v2 detection model
Results:
Detection on an image with ssdlite_mobilnet_v2

![image](https://github.com/user-attachments/assets/318b17d3-4a7a-439b-a5be-d4a2692a2d41)

The image shows:

89% chance that the detected ingredient is an orange
97% chance that the detected ingredient is a banana
40% chance that the detected ingredient is an apple.
c. Bird Song Recognition
The Bird Song Dataset on Kaggle is a collection of bird songs, often used for audio classification projects and machine learning. This dataset contains audio recordings of bird songs from various species, making it an excellent tool for researchers and developers working on audio data analysis tasks, such as species identification by their song.

Techniques used:
Calculate the spectrogram via STFT (Short-Term Fourier Transform)
Convert the spectrogram to decibels
Normalize the values and provide the normalized spectrogram with the associated label
Convolutional Neural Network
Results:
Model evaluation

![image](https://github.com/user-attachments/assets/8d4de32f-fed4-4051-9dd7-7b807e8adb5d)


Confusion Matrix

![image](https://github.com/user-attachments/assets/f2c02ac9-1137-4ee6-afd9-edf934dba314)

Score:

F1: 0.9510736108330955 | Precision: 0.9504173755284869 | Recall: 0.9524755155991809 | AUC: 0.9979307755194758

## 4. Large Language Model (LLM)
### a. Tweet Reviews (NLP)
The goal is to determine the sentiment of individuals based on their tweets about U.S. airlines. The dataset contains 14,640 tweets, extracted from Twitter data in February 2015. In this case, we use an LLM (BERT) algorithm with Hugging Face to generate a score from 1 to 5 indicating the sentiment of the tweet.

Results:
![9migpjpi](https://github.com/user-attachments/assets/8b10c5c9-ebd1-4cca-aefb-91c7761515b8)

### b. Using RAG with the LLM model (LLaMa2)
RAG (Retrieval-Augmented Generation) combines information retrieval and text generation to enhance the accuracy of responses by leveraging external data. By integrating relevant documents before the generation phase, it reduces hallucinations and provides more factual and up-to-date answers. I performed a test using my professional thesis PDF on the study and implementation of interpretability methods in car pricing models.

## 5. API Development
### a. Restaurant Location and Recommendation
In this project, I used MongoDB and Python to work with a popular New York restaurant database. This database includes information such as the name, address, cuisine type, geographical location (latitude and longitude coordinates), and restaurant ratings, including scores and grades given by customers. I developed an interactive map to visualize restaurant locations and integrated dynamic filters to allow users to refine their search based on cuisine type, maximum score, and restaurant grade.

![image](https://github.com/user-attachments/assets/49c9096d-ed43-44cc-b6a1-ef771e3c3ec5)



![image](https://github.com/user-attachments/assets/aca78036-a8f7-486a-ac68-ac57786d6636)

### b. Iris Dataset
The Iris dataset is one of the most famous datasets in machine learning. It contains 150 observations of flowers belonging to three different species: Iris-setosa, Iris-versicolor, and Iris-virginica. Each observation is described by 4 features:

- Sepal length (in cm)
- Sepal width (in cm)
- Petal length (in cm)
- Petal width (in cm)
The goal is often to predict the species of the flower based on these features. This dataset is commonly used for classification tasks and to test different supervised learning algorithms.

![image](https://github.com/user-attachments/assets/3d7d3aa0-2220-45f7-b075-87b7b25a97c5)

## 6. Optimization Projects
- Construction of a stadium within the shortest possible time. (cplex/python)
- Graph theory: Dijkstra's algorithm, nearest neighbor.
- Advanced optimization: Heuristic for the traveling salesman problem, stochastic optimization (Engie: renewable electricity mix in 2050).

