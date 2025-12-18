import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Charger les données
df = pd.read_csv('ventes_fictives.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Nettoyage : Supprimer les valeurs manquantes (si présentes)
df.dropna(inplace=True)

# Analyse 1 : Tendances des ventes mensuelles
df['Mois'] = df['Date'].dt.to_period('M')
ventes_mensuelles = df.groupby('Mois')['Total_Vente'].sum()
plt.figure(figsize=(10, 5))
ventes_mensuelles.plot(kind='line', marker='o')
plt.title('Tendances des Ventes Mensuelles')
plt.xlabel('Mois')
plt.ylabel('Total Ventes (€)')
plt.grid(True)
plt.savefig('tendances_ventes.png')
plt.show()

# Analyse 2 : Produits les plus vendus
produits_populaires = df.groupby('Produit')['Quantite'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
produits_populaires.plot(kind='bar')
plt.title('Top 10 Produits par Quantité Vendue')
plt.xlabel('Produit')
plt.ylabel('Quantité')
plt.savefig('produits_populaires.png')
plt.show()

# Analyse 3 : Ventes par région
ventes_region = df.groupby('Region')['Total_Vente'].sum()
plt.figure(figsize=(8, 8))
ventes_region.plot(kind='pie', autopct='%1.1f%%')
plt.title('Répartition des Ventes par Région')
plt.savefig('ventes_region.png')
plt.show()

# Data Mining : Clustering des clients (par total dépensé et nombre d'achats)
client_data = df.groupby('Client_ID').agg({'Total_Vente': 'sum', 'Vente_ID': 'count'}).rename(columns={'Vente_ID': 'Nombre_Achats'})
scaler = StandardScaler()
client_scaled = scaler.fit_transform(client_data)
kmeans = KMeans(n_clusters=3, random_state=42)
client_data['Cluster'] = kmeans.fit_predict(client_scaled)
sns.scatterplot(data=client_data, x='Total_Vente', y='Nombre_Achats', hue='Cluster', palette='viridis')
plt.title('Segmentation des Clients (Clustering K-Means)')
plt.xlabel('Total Dépensé (€)')
plt.ylabel('Nombre d\'Achats')
plt.savefig('clustering_clients.png')
plt.show()

# Insights : Afficher des statistiques
print("Statistiques générales :")
print(df.describe())
print("\nTotal des ventes :", df['Total_Vente'].sum())
print("Produit le plus populaire :", produits_populaires.index[0])