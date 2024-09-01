#Importation des packages

from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Initialison de Flask
app = Flask(__name__)

# Charger le modèle et les outils de prétraitement
model = joblib.load('best_model.pkl')


# Route pour la page web
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si la requête contient des données JSON
    data = request.get_json()
    
    if not all(k in data for k in ('sepal_length', 'sepal_width', 'petal_length', 'petal_width')):
        return jsonify({'error': 'Missing input data'}), 400
    
    # Extraire les caractéristiques du JSON
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    # Préparer les données pour la prédiction : mettre sous forme de tableau
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Faire la prédiction
    prediction = model.predict(features)
    
    # Charger les noms des classes
    iris = load_iris()
    class_names = iris.target_names
    predicted_class = class_names[prediction[0]]
    
    # Retourner la réponse JSON
    return jsonify({'predicted_class': predicted_class})

# Exécuter l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
