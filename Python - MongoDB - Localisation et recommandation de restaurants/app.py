from flask import Flask, jsonify, request, render_template
from pymongo import MongoClient, errors
import folium
from folium.plugins import MarkerCluster
import requests

app = Flask(__name__)

# Connexion à la base de données MongoDB
try:
    # Connexion à MongoDB sur le port par défaut
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Test"]  # Sélection de la base de données "Test"
    client.admin.command('ping')  # Tester la connexion à MongoDB
    print("Connexion à MongoDB réussie !")
except errors.ConnectionFailure:
    # Si la connexion échoue, afficher une erreur et arrêter l'application
    print("Erreur de connexion à MongoDB !")
    exit(1)

# Définir la collection 'restaurants' dans la base de données
restaurants_collection = db["restaurants"]

@app.route("/restaurants", methods=["GET"])
def get_restaurants():
    # Récupération des paramètres de filtrage depuis l'URL
    cuisine_type = request.args.get('cuisine_type', default=None, type=str)
    max_score = request.args.get('max_score', default=None, type=int)  
    grade_filter = request.args.get('grade', default=None, type=str)

    # Construire la requête pour MongoDB en fonction des filtres appliqués
    query = {}
    if cuisine_type:
        query["cuisine"] = cuisine_type  # Filtrer par type de cuisine
    if max_score is not None:
        query["grades.score"] = {"$lte": max_score}  # Filtrer par score maximum
    if grade_filter:
        query["grades.grade"] = grade_filter  # Filtrer par grade

    # Effectuer la recherche dans la collection de restaurants avec les critères de filtrage
    restaurants = restaurants_collection.find(query)

    # Préparer la liste des restaurants sous forme de dictionnaire
    result = []
    for restaurant in restaurants:
        restaurant_data = {
            "name": restaurant.get("name"),
            "address": restaurant.get("address", {}).get("building", "") + " " + restaurant.get("address", {}).get("street", ""),
            "cuisine_type": restaurant.get("cuisine", ""),
            "location": restaurant.get("address", {}).get("coord", []),
            "grades": restaurant.get("grades", [])
        }
        result.append(restaurant_data)  # Ajouter les données formatées à la liste des résultats

    return jsonify(result)  # Retourner la liste des restaurants sous forme de JSON

@app.route("/", methods=["GET"])
def index():
    # Récupérer les filtres pour la carte depuis l'URL
    cuisine_type = request.args.get('cuisine_type', default=None, type=str)
    max_score = request.args.get('max_score', default=None, type=int)
    grade_filter = request.args.get('grade', default=None, type=str)

    # URL de l'API pour récupérer les restaurants
    API_URL = "http://127.0.0.1:5000/restaurants"

    # Appeler l'API pour récupérer les restaurants avec les filtres
    response = requests.get(API_URL, params={"cuisine_type": cuisine_type, "max_score": max_score, "grade": grade_filter})
    restaurants = response.json()  # Obtenir les données de l'API sous forme de JSON

    # Créer une carte centrée sur New York
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # Carte centrée sur New York
    marker_cluster = MarkerCluster().add_to(m)  # Ajouter un cluster de marqueurs pour les restaurants

    # Définir les couleurs pour chaque type de cuisine
    cuisine_colors = {
        "Bakery": "#DA70D6",
        "Hamburgers": "#FFFF00",
        "Chinese": "#00FFFF",
        "Italian": "#32CD32",
        "Mexican": "#FF00FF",
        "Autre": "#000000"
    }

    # Fonction pour vérifier si un restaurant est situé dans la zone de New York
    def is_in_new_york(lat, lon):
        # Limites géographiques approximatives de New York
        return 40.4774 <= lat <= 40.9176 and -74.2591 <= lon <= -73.7004

    # Ajouter un marqueur pour chaque restaurant sur la carte
    for restaurant in restaurants:
        name = restaurant["name"]
        address = restaurant["address"]
        cuisine = restaurant["cuisine_type"]
        location = restaurant["location"]
        grades = restaurant["grades"]

        # Calculer le score moyen à partir des grades
        if grades:
            avg_score = sum(grade["score"] for grade in grades) / len(grades)
        else:
            avg_score = "N/A"  # Si aucun score n'est disponible, afficher "N/A"

        # Créer l'infobulle avec les informations du restaurant
        popup_info = f"<b>Nom:</b> {name}<br><b>Adresse:</b> {address}<br><b>Cuisine:</b> {cuisine}<br><b>Score moyen:</b> {avg_score}"

        if location and len(location) == 2:  # Vérifier si les coordonnées sont disponibles
            lon, lat = location

            # Vérifier si les coordonnées sont dans la zone de New York
            if is_in_new_york(lat, lon):
                # Déterminer la couleur du marqueur selon le type de cuisine
                color = cuisine_colors.get(cuisine, cuisine_colors["Autre"])

                # Ajouter un marqueur sur la carte avec les détails du restaurant
                folium.CircleMarker(
                    location=[lat, lon],  # Positionner le marqueur à la latitude et longitude
                    radius=6,  # Taille du marqueur
                    popup=popup_info,  # Ajouter l'infobulle au marqueur
                    color=color,  # Couleur du marqueur
                    fill=True,
                    fill_color=color,  # Remplir le marqueur avec la même couleur
                    fill_opacity=0.7  # Opacité du remplissage
                ).add_to(marker_cluster)  # Ajouter le marqueur au cluster

    # Sauvegarder la carte dans le dossier 'static'
    map_html = "static/restaurants_map.html"  # Chemin où la carte sera sauvegardée
    m.save(map_html)  # Sauvegarder la carte sous forme de fichier HTML

    # Rendre le template HTML avec la carte
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)  # Lancer l'application en mode debug




#Pour vérifier la base de données collectée sur MongoDB. : http://127.0.0.1:5000/restaurants

#Pour lancer l'application : http://127.0.0.1:5000/
