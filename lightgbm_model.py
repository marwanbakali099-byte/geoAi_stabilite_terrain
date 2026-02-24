import os
import webbrowser

import folium
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# 1-Exploiatation des données et etude des données
df = pd.read_csv("geo_stabilite_terrain_data.csv")
print(df.head(10))
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())


# 3-Prétraitement des données et Gestion des valeurs manque 
df["stabilite_terrain"] = df["stabilite_terrain"].str.lower().str.strip()

df["stabilite_terrain"] = df["stabilite_terrain"].replace({
    "moyennement_stable": "moyen",
    "moyennement stable": "moyen",
    "moy_stable": "moyen",
    "stable": "stable",
    "instable": "instable"
})

df.drop(columns="indice_geotech_labo",inplace=True)
df["pente_pct"] = df["pente_pct"].fillna(df["pente_pct"].mean())
df["altitude_m"] = df["altitude_m"].fillna(df["altitude_m"].mode()[0])
df["texture_sol"] = df["texture_sol"].fillna(df["texture_sol"].mean())
df["humidite_sol"] = df["humidite_sol"].fillna(df["humidite_sol"].mean())
df["distance_faille_km"] = df["distance_faille_km"].fillna(df["distance_faille_km"].mean())
df["couverture_vegetale"] = df["couverture_vegetale"].fillna(df["couverture_vegetale"].mean())
df.drop(1026,inplace=True)


# 4-1-Features et target de l'entrainement
X = df[["pente_pct","altitude_m","texture_sol","humidite_sol","distance_faille_km","couverture_vegetale"]]
Y = df["stabilite_terrain"]

# 4-2-Standarisation des données
stand = StandardScaler()
X_stand = stand.fit_transform(X)

# 4-3-Séparation des données
X_train,X_test,Y_train,Y_test = train_test_split(X_stand, Y, test_size=0.25, random_state=42)

# 5-1-Création de la modéle par la methode lightgbm
 
model = LGBMClassifier(
    objective="multiclass",
    num_class=3,  # nombre de classes
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

# 5-2-Entrainement de la modèle 
model.fit(X_train,Y_train)

# 5-3-Prédéction de la modèle testing
Y_pred = model.predict(X_test)

# 6-Evaluation de modéle
print("-----------Evaluation------------")
print("Accuracy (Exactitude): ",accuracy_score(Y_test,Y_pred))
print("Martice de confussion")
print(confusion_matrix(Y_test,Y_pred))
print("rapport de classification")
print(classification_report(Y_test,Y_pred))


conf_matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Prédit Instable', 'Prédit Moyen', 'Prédit Stable'],
    yticklabels=['Réel Instable', 'Réel Moyen', 'Réel Stable']
)

plt.xlabel('Prédiction du Modèle')
plt.ylabel('Réalité du Terrain')
plt.title('Matrice de Confusion - Stabilité des Terrains')
plt.show()

# Prédiction sur tout le dataset
df["prediction_rf"] = model.predict(X_stand)

print(df[["longitude", "latitude", "stabilite_terrain", "prediction_rf"]].head())

# Centre approximatif (Nord du Maroc)
m = folium.Map(location=[35.0, -5.9], zoom_start=8)

# Couleurs selon la classe prédite
colors = {
    "instable": "red",
    "moyen": "orange",
    "stable": "green"
}

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=colors[row["prediction_rf"]],
        fill=True,
        fill_opacity=0.7,
        popup=f"""
        <b>Zone ID:</b> {row["zone_id"]}<br>
        <b>Réel:</b> {row["stabilite_terrain"]}<br>
        <b>Prédit (RF):</b> {row["prediction_rf"]}
        """
    ).add_to(m)

# Sauvegarde de la carte
m.save("carte_random_forest.html")
webbrowser.open(os.path.abspath("carte_random_forest.html"))