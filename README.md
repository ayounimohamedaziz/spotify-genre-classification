# 🎵 Spotify Track Genre Classification

Projet de Machine Learning visant à prédire le genre musical d'un morceau Spotify parmi 114 catégories en combinant ses caractéristiques audio numériques et des métadonnées textuelles (*artistes*, *albums*, *titres*).

---

## 📊 Résultats & Performances

* **Métrique d'évaluation :** Accuracy sur le jeu de validation (20% stratifié)
* **Score obtenu :** **48.30 %** (sur 114 classes d'équilibre équivalent)

---

## 🛠️ Stack Technique

* **Langage :** Python 3
* **Manipulation de données :** Pandas, NumPy
* **Traitement & Preprocessing :** Scikit-Learn (`TfidfVectorizer`, `ColumnTransformer`, `Pipeline`, `LabelEncoder`)
* **Modélisation :** XGBoost (`XGBClassifier` avec accélération GPU `cuda` / `tree_method='hist'`)

---

## 💡 Architecture de la Solution

### 1. Préparation des Données & Feature Engineering
* **Nettoyage :** Suppression des lignes contenant des valeurs manquantes critiques (`artists`, `track_genre`).
* **Feature Textuelle Globale (`text_info`) :** Fusion des colonnes `artists`, `album_name` et `track_name` pour capturer l'ensemble du contexte sémantique de la piste.
* **Conversion :** Transformation de la colonne booléenne `explicit` en valeur numérique (`int`).

### 2. Pipeline de Preprocessing (`ColumnTransformer`)
* **Variables Numériques (15 au total) :** Conservation brute via `passthrough` (`popularity`, `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`).
* **Variable Textuelle (`text_info`) :** Vectorisation par `TfidfVectorizer` :
  * `max_features = 4500`
  * `ngram_range = (1, 2)` (mots simples et bigrammes)
  * `stop_words = 'english'`

### 3. Modélisation XGBoost
Entraînement d'un modèle `XGBClassifier` optimisé avec les hyperparamètres suivants :
* `n_estimators = 400`
* `learning_rate = 0.07`
* `max_depth = 6`
* `subsample = 0.8`
* `colsample_bytree = 0.8`
* `tree_method = 'hist'` & `device = 'cuda'`

---

## 🚀 Structure du Code

1. **Chargement et inspection :** Lecture de `dataset.csv` et vérification des types et valeurs manquantes.
2. **Encodage de la cible :** Conversion de `track_genre` en étiquettes numériques via `LabelEncoder`.
3. **Pipeline Scikit-Learn :** Enchaînement du prétraitement et de la classification.
4. **Validation :** Découpage 80/20 avec stratification (`stratify=y`) pour conserver l'équilibre des classes.
