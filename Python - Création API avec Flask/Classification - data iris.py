import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

import os
os.chdir(r'C:\Users\flori\OneDrive\Bureau\all\Ecole\Ingénieur\Data_science-main\Data_science-main\Python - Création API avec Flask')

# Charger les données Iris
# Remarque : Cet ensemble de données est bien connu pour les démonstrations de classification
iris = load_iris()
X = iris.data  # Les caractéristiques (features)
y = iris.target  # Les étiquettes (labels)

# 2. Préparer les données
# Les données sont déjà prêtes avec X et y

#On crée 3 modèles de classification :  Initialiser les modèles de classification
# On choisit ces modèles car ils sont connus pour leurs performances dans les scénarios de classification
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear',probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

#On appliquer la validation croisée pour avoir un modèle robuste
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)  # Utilisation de la validation croisée avec 5 phases d'entrainement 
    results[name] = cv_scores
    print(f"{name} - Mean Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# On détermine le meilleur modèle
best_model_name = max(results, key=lambda k: np.mean(results[k]))
best_model = models[best_model_name]
print(f"\nLe meilleur modèle est {best_model_name} avec une précision moyenne de {np.mean(results[best_model_name]):.4f}")

# On Entraîner le meilleur modèle sur l'ensemble complet des données
best_model.fit(X, y)

# On sauvegarde le meilleur modèle
model_filename = 'best_model.pkl'
joblib.dump(best_model, model_filename)
print(" Le modèle est sauvegardé")
