# 📊 Analyse des Ventes - Python & Power BI

Ce projet présente une analyse détaillée des ventes à travers différents tableaux de bord interactifs.

## 🖥️ Aperçu des Dashboards

### 1️⃣ Dashboard Générale
Vue d'ensemble de toutes les activités de vente.
![Dashboard Generale](./dashboard-generale.png)

---

### 2️⃣ Dashboard des Vêtements
Analyse spécifique au secteur de l'habillement.
![Dashboard des Vêtements](./dashboard-vetements.png)

---

### 3️⃣ Dashboard des Électronique
Suivi des performances des produits électroniques.
![Dashboard des Electronique](./dashboard-electronique.png)

## 📂 Contenu du Projet
- **`analyse_ventes.py`** : Script Python pour le traitement des données.
- **`analyse des ventes.pbix`** : Fichier Power BI.
- **`ventes_fictives.csv`** : Jeu de données utilisé.

## Description
Ce projet est une analyse complète des données de ventes d'une boutique en ligne fictive sur l'année 2023. Il couvre le nettoyage des données, les visualisations, l'analyse de tendances, et du data mining simple (clustering des clients). L'objectif est de démontrer des compétences en data analysis, data mining et visualisation avec Python et Power BI.

## Structure du Projet
- `ventes_fictives.csv` : Dataset fictif avec 500 ventes (colonnes : Date, Produit, Quantité, Prix, Région, etc.).
- `analyse_ventes.py` : Script Python pour nettoyer, analyser et générer des visualisations (tendances mensuelles, produits populaires, ventes par région, clustering clients).
- `rapport_analyse.md` : Rapport détaillé des résultats et insights.
- `analyse des ventes.pbix` : Dashboard interactif Power BI (ouvre avec Power BI Desktop).
- Images générées : `tendances_ventes.png`, `produits_populaires.png`, `ventes_region.png`, `clustering_clients.png`.

## Comment Exécuter
1. **Prérequis** : Installe Python (avec pip) et les bibliothèques : `pip install pandas matplotlib seaborn scikit-learn`.
2. Télécharge ou clone ce repo.
3. Exécute le script Python : `python analyse_ventes.py` (il génère les images PNG automatiquement).
4. Ouvre `analyse des ventes.pbix` dans Power BI Desktop pour explorer le dashboard interactif.
5. Lis le `rapport_analyse.md` pour les insights détaillés.

## Compétences Démontrées
- **Data Cleaning** : Nettoyage et préparation des données avec pandas.
- **Data Analysis** : Analyses descriptives (tendances temporelles, produits populaires, ventes par région).
- **Data Mining** : Clustering K-Means pour segmenter les clients en groupes (bas, moyen, haut dépensiers).
- **Visualisations** : Graphiques avec matplotlib et seaborn.
- **Dashboard Interactif** : Création de rapports dynamiques avec Power BI.

## Auteur
[MEZGHIT ILIAS] - Étudiant en 2ème année Master Ingénierie de la Décision.  
LinkedIn : [[ILIAS MEZGHIT](https://www.linkedin.com/in/ilias-mezghit/)]  
Email : [mezghit.ilias.feg_emid@uhp.ac.ma]

## Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.