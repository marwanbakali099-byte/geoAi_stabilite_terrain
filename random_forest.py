import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import os
import webbrowser

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
df = df.dropna(subset=["stabilite_terrain"])
print(df.isnull().sum())


# 4-1-Features et target de l'entrainement
X = df[["pente_pct","altitude_m","texture_sol","humidite_sol","distance_faille_km","couverture_vegetale"]]
Y = df["stabilite_terrain"]

# 4-2-Standarisation des données
stand = StandardScaler()
X_stand = stand.fit_transform(X)

# 4-3-Séparation des données
X_train,X_test,Y_train,Y_test = train_test_split(X_stand, Y, test_size=0.25, random_state=42)

# 5-1-entrainement par la methode Random Forest
model_rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
model_rf.fit(X_train,Y_train)

# 5-2-Prédiction de modéle testing  
Y_pred = model_rf.predict(X_test)

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
df["prediction_rf"] = model_rf.predict(X_stand)

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

print(model_rf.score(X_train,Y_train))
print(model_rf.score(X_test, Y_test))
