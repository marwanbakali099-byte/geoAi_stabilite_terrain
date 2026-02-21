import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 1-Exploiatation et étudiant des données 
df = pd.read_csv("geo_stabilite_terrain_data.csv")
print(df.head(10))
print(df.info())
print(df.shape)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())

# 2-Prétraitement des données et Gestion des valeurs manques
df["stabilite_terrain"] = df["stabilite_terrain"].str.lower().str.strip()
df["stabilite_terrain"] = df["stabilite_terrain"].replace({
    "moyennement_stable":"moyen_stable",
    "moyennement stable":"moyen_stable",
    "moy_stable":"moyen_stable",
    "stable":"stable",
    "instable":"instable"
})
df.drop(columns='indice_geotech_labo',inplace=True)
df["pente_pct"] = df["pente_pct"].fillna(df["pente_pct"].mean())
df["altitude_m"] = df["altitude_m"].fillna(df["altitude_m"].mode()[0])
df["texture_sol"] = df["texture_sol"].fillna(df["texture_sol"].mean())
df["humidite_sol"] = df["humidite_sol"].fillna(df["humidite_sol"].mean())
df["distance_faille_km"] = df["distance_faille_km"].fillna(df["distance_faille_km"].mean())
df["couverture_vegetale"] = df["couverture_vegetale"].fillna(df["couverture_vegetale"].mean())
df.drop(1026,inplace=True)
print(df.isnull().sum())


# # 3-visualisation des données
# sns.countplot(x='stabilite_terrain' ,data=df)
# plt.title("destribution de la stabilite des terrains")
# # plt.show()

# 3-Featurs et target
X = df[["pente_pct","altitude_m","texture_sol","humidite_sol","distance_faille_km",]]
Y = df["stabilite_terrain"]

# 4-Normalisation des données
print(X)
stand = StandardScaler()
X_stand = stand.fit_transform(X)
print(X_stand)

# 5-Séparation des données
X_train,X_test,Y_train,Y_test = train_test_split(X_stand,Y,test_size=0.25,random_state=42)

# # 6-1-Entrainement par la methode SVM
# model = SVC(kernel="linear", decision_function_shape="ovo")
# model.fit(X_train,Y_train)
# # Paramètres à tester
# param_grid = [
#     {
#         'kernel': ['linear'],
#         'C': [0.1, 1, 10, 100]
#     },
#     {
#         'kernel': ['rbf'],
#         'C': [0.1, 1, 10, 100],
#         'gamma': [0.1, 1, 10, 100]
#     }
# ]

# # Création du GridSearch
# grid_search = GridSearchCV(
#     SVC(random_state=42),  # Votre modèle SVM
#     param_grid,           # Les paramètres à tester
#     cv=5,                 # 5-fold validation croisée
#     scoring='accuracy',   # Métrique à optimiser
#     n_jobs=-1             # Utilise tous les processeurs
# )

# # Exécution
# grid_search.fit(X_train, Y_train)

# # 6-2-Prédéction de modéle 
# Y_pred = grid_search.predict(X_test)

# # 7-Evaluation de modéle
# print("-----------Evaluation------------")
# print("Accuracy (Exactitude): ",accuracy_score(Y_test,Y_pred))
# print("Martice de confussion")
# print(confusion_matrix(Y_test,Y_pred))
# print("rapport de classification")
# print(classification_report(Y_test,Y_pred))
# conf_matrix = confusion_matrix(Y_test, Y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     conf_matrix,
#     annot=True,
#     fmt='d',
#     cmap='Greens',
#     xticklabels=['Prédit Instable', 'Prédit Moyen', 'Prédit Stable'],
#     yticklabels=['Réel Instable', 'Réel Moyen', 'Réel Stable']
# )
# plt.xlabel('Prédiction du Modèle')
# plt.ylabel('Réalité du Terrain')
# plt.title('Matrice de Confusion - Stabilité des Terrains')
# plt.show()

# 6-1-Optimisation et entraînement avec GridSearchCV
param_grid = [
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    },
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 10, 100]
    }
]

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1  # Ajoutez ceci pour voir la progression
)

print("Début de l'optimisation GridSearchCV...")
grid_search.fit(X_train, Y_train)

# Afficher les résultats de GridSearchCV
print("\n" + "="*50)
print("RÉSULTATS GRIDSEARCHCV")
print("="*50)
print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score (validation croisée): {grid_search.best_score_:.3f}")
print(f"Nombre de combinaisons testées: {len(grid_search.cv_results_['params'])}")

# Obtenir le meilleur modèle
best_model = grid_search.best_estimator_
print(f"\nModèle sélectionné: {best_model}")

# 6-2-Prédiction avec le meilleur modèle
Y_pred = best_model.predict(X_test)

# 7-Évaluation du modèle
print("\n" + "="*50)
print("ÉVALUATION SUR LE JEU DE TEST")
print("="*50)
print(f"Accuracy (Exactitude): {accuracy_score(Y_test, Y_pred):.3f}")
print("\nMatrice de confusion:")
print(confusion_matrix(Y_test, Y_pred))
print("\nRapport de classification:")
print(classification_report(Y_test, Y_pred))

# Visualisation de la matrice de confusion
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
plt.title('Matrice de Confusion - Stabilité des Terrains (Modèle Optimisé)')
plt.show()